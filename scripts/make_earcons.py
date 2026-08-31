#!/usr/bin/env python
"""Bake the earcon lexicon to WAVs, and audition it.

The app renders each tone lazily on first use, so this script is never
required; it exists for the thing the proposal asked for before a single
event subscriber was wired -- rendering all six back to back and LISTENING
to them, to hear whether they are one family or six unrelated beeps.

    ~/vss_env/bin/python scripts/make_earcons.py                # bake into MEMORY_DIR
    ~/vss_env/bin/python scripts/make_earcons.py --out /tmp/ec  # bake elsewhere
    ~/vss_env/bin/python scripts/make_earcons.py --out /tmp/ec --play

--play walks the lexicon with a second between tones, through the same
paplay -> aplay chain the app uses. Nothing here touches the live app, the
running config or the assistant's state: with --out it writes only where
you point it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import earcons  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="", help="directory to write into "
                                              "(default: PATHS.MEMORY_DIR/earcons)")
    ap.add_argument("--play", action="store_true", help="play each tone after baking")
    ap.add_argument("--gap", type=float, default=1.0, help="seconds between tones")
    args = ap.parse_args()

    out = Path(args.out).expanduser() if args.out else None
    # force=True: an audition wants THIS run's spec, not last week's cache.
    made = earcons.render_all(out, force=True)
    for name in earcons.NAMES:
        path = made.get(name)
        if path is None:
            print(f"  {name:<10} FAILED")
            continue
        notes, gain = earcons.TONES[name]
        shape = " -> ".join(f"{int(f)}Hz/{int(ms)}ms" for f, ms in notes)
        print(f"  {name:<10} {path}  [{shape}, gain {gain}]")
    if not args.play:
        return 0
    for name in earcons.NAMES:
        path = made.get(name)
        if path is None:
            continue
        print(f"playing {name}")
        for argv in (["paplay", str(path)], ["aplay", "-q", str(path)]):
            try:
                subprocess.run(argv, check=False, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue
        time.sleep(max(0.0, args.gap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
