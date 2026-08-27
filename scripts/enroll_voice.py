#!/usr/bin/env python3
"""Enroll your voice so Jarvis can tell you apart from the room.

Speaker verification is already built (`jarvis/speaker.py`) and already wired
into the capture path (`jarvis/app.py:763`, `jarvis/recorder.py:591`), but it
is inert until a voiceprint exists: with zero embeddings `verify()` fails
OPEN and accepts every voice, which is why a TV can reach the commander.

This records several takes through the SAME path verification uses --
`Recorder.record_fixed`, which opens the mic at its native rate and resamples
to 16 kHz with scipy. Channel match matters: a voiceprint captured through a
different mic or resampler is the usual cause of false rejects later.

Each take is scored against the running centroid before it is kept, so a bad
take (mic bumped, someone else talking, too quiet) is visible immediately
instead of quietly dragging the centroid off.

Usage:
    ~/vss_env/bin/python scripts/enroll_voice.py            # 6 takes x 8s
    ~/vss_env/bin/python scripts/enroll_voice.py --takes 8 --seconds 10
    ~/vss_env/bin/python scripts/enroll_voice.py --status   # show current state
    ~/vss_env/bin/python scripts/enroll_voice.py --reset    # discard and start over
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.config import CONFIG, PATHS          # noqa: E402
from jarvis.recorder import MicArbiter, Recorder  # noqa: E402
from jarvis.speaker import SpeakerVerifier        # noqa: E402

JARVIS_PID = Path("/tmp/vss_voice/jarvis.pid")

# Varied prompts beat one long monotone take: ECAPA compares against the
# centroid of all embeddings, so a spread of natural phrasing and distance
# generalises better than six readings of the same sentence.
PROMPTS = [
    ("Speak normally, from where you usually sit.",
     "Hey Jarvis, what's on my calendar today?"),
    ("Same spot, normal voice.",
     "Open the terminal and check the build status for me."),
    ("A little quieter, like it's late.",
     "Turn the volume down and read me the last message."),
    ("Lean back or sit a bit further away.",
     "Jarvis, remind me to call the supplier at four o'clock."),
    ("Normal again, but say it a bit faster.",
     "What's the weather looking like this afternoon?"),
    ("Relaxed, conversational -- like you actually talk to him.",
     "Alright, go ahead and summarise where we left off yesterday."),
    ("One more, normal distance.",
     "Show me the detection results from the last run."),
    ("Last one -- whatever you want to say.",
     "(say anything, about eight seconds' worth)"),
]

MIN_RMS = 0.004        # below this the take is effectively silence
LOW_SCORE = 0.55       # a take this far from the centroid is worth redoing


def show_status(v: SpeakerVerifier) -> None:
    print(f"voiceprint file : {PATHS.VOICEPRINT}")
    print(f"exists          : {PATHS.VOICEPRINT.exists()}")
    print(f"samples enrolled: {v.num_samples}")
    print(f"threshold       : {v.threshold}")
    print(f"speaker_verify  : {CONFIG.speaker_verify}")
    if v.num_samples >= 2:
        sims = [v._cosine_similarity(e, v._centroid) for e in v._embeddings]
        print(f"self-consistency: min={min(sims):.3f} mean={np.mean(sims):.3f}")
        print("  (min well below the others means one take is an outlier)")


def record_take(rec: Recorder, seconds: float, label: str, line: str) -> np.ndarray:
    print(f"\n  {label}")
    print(f'  Say: "{line}"')
    for n in (3, 2, 1):
        print(f"    {n}...", end="", flush=True)
        time.sleep(1)
    print(f" recording {seconds:.0f}s -- go.")
    audio = rec.record_fixed(seconds)
    print("    done.")
    return audio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--status", action="store_true", help="print state and exit")
    ap.add_argument("--reset", action="store_true", help="discard the existing voiceprint first")
    ap.add_argument("--yes", action="store_true", help="never prompt to redo a take")
    args = ap.parse_args()

    verifier = SpeakerVerifier(gpu=0, threshold=CONFIG.speaker_threshold)
    verifier.load()

    if args.status:
        show_status(verifier)
        return 0

    if JARVIS_PID.exists():
        print("NOTE: Jarvis appears to be running. It holds the mic and its wake")
        print("      word will fire on your enrolment speech. Stop it first for a")
        print("      clean run, or expect some noise in its log.\n")

    print("Loading the ECAPA-TDNN speaker model...")
    if not verifier.load_model():
        print("ERROR: speaker model failed to load; cannot enrol.", file=sys.stderr)
        return 1

    if args.reset and verifier.num_samples:
        verifier.clear()
        print("Cleared the existing voiceprint.")

    rec = Recorder(MicArbiter(), speaker_verifier=None)
    if not rec.mic_available:
        print("ERROR: no microphone detected.", file=sys.stderr)
        return 1

    print(f"\nEnrolling {args.takes} takes of {args.seconds:.0f}s "
          f"(starting from {verifier.num_samples} existing samples).")
    print("Speak as you normally would to Jarvis -- this is what he'll match against.")

    kept = 0
    take = 0
    while kept < args.takes:
        label, line = PROMPTS[take % len(PROMPTS)]
        take += 1
        audio = record_take(rec, args.seconds, f"[{kept + 1}/{args.takes}] {label}", line)

        if audio is None or len(audio) == 0:
            print("    no audio captured -- retrying.")
            continue

        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < MIN_RMS:
            print(f"    too quiet (rms={rms:.4f}) -- mic muted or wrong device? retrying.")
            continue

        # Score against what we already have BEFORE committing, so an outlier
        # never enters the centroid it is being measured against.
        if verifier.num_samples:
            _, score = verifier.verify(audio)
            note = "  <-- unlike your other takes" if score < LOW_SCORE else ""
            print(f"    rms={rms:.4f}  similarity to your voiceprint={score:.3f}{note}")
            if score < LOW_SCORE and not args.yes:
                if input("    keep it anyway? [y/N] ").strip().lower() != "y":
                    print("    discarded -- let's redo that one.")
                    continue
        else:
            print(f"    rms={rms:.4f}  (first take -- nothing to compare against yet)")

        ok, total = verifier.enroll_from_audio(audio)
        if not ok:
            print("    embedding extraction failed -- retrying.")
            continue
        kept += 1
        print(f"    kept. {total} sample(s) enrolled.")

    print("\n--- enrolment complete ---")
    show_status(verifier)
    print("\nNext: measure a threshold with scripts/tune_speaker_threshold.py")
    print("before turning speaker_verify on -- the 0.40 default is a guess.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
