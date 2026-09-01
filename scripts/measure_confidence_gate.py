#!/usr/bin/env python3
"""Re-measure the transcript confidence gate against a real session log.

``jarvis/transcriber.py`` rejects a transcript whose mean segment
avg_logprob is below ``MIN_AVG_LOGPROB``. That number was originally set
from an impression ("genuine utterances on this mic score -0.2..-0.6"),
and on 2026-08-31 it cost Hunter nine features in one evening: every one
of "belay that" (-0.88), "scratch that" (-0.89), "volume 40" (-0.87),
"Play my liked songs." (-0.97), "max volume" (-2.08) and a plain "Yes."
(-0.95) answering Jarvis's own read-back was transcribed CORRECTLY and
thrown away. Whisper's avg_logprob is length-biased -- a short clip has
few tokens, so the end-of-text token dominates the mean -- so the gate
was worst at exactly the utterances an answer and a command look like.

Do not move MIN_AVG_LOGPROB without running this. It reads the
``Transcribed: <text> (avg_logprob=<n>)`` lines a session leaves in
/tmp/vss_voice/jarvis.log, prints them ranked, and reports how many
utterances each candidate threshold would reject. Reading the ranked
list is the point: the machine cannot tell you which lines are real, but
the two populations are obvious by eye (real commands at the top,
character salad and repeated-token loops at the bottom).

    python scripts/measure_confidence_gate.py [logfile] [--band LO HI]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

DEFAULT_LOG = Path("/tmp/vss_voice/jarvis.log")
LINE_RX = re.compile(r"Transcribed: (.*) \(avg_logprob=(-?[0-9.]+)\)")
# The thresholds worth comparing: the original, the one that caused the
# 2026-08-31 losses, and the one measured to replace it.
CANDIDATES = (-0.85, -1.50, -2.90, -3.50)


def read(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open(errors="replace") as fh:
        for line in fh:
            m = LINE_RX.search(line)
            if not m:
                continue
            raw = m.group(1)
            try:
                text = ast.literal_eval(raw)     # the log writes %r
            except (ValueError, SyntaxError):
                text = raw
            rows.append((str(text), float(m.group(2))))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logfile", nargs="?", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--band", nargs=2, type=float, metavar=("LO", "HI"),
                    default=(-2.90, -0.85),
                    help="print only this score band in full (default: the "
                         "band the old gate threw away)")
    args = ap.parse_args(argv)

    if not args.logfile.exists():
        print(f"no such log: {args.logfile}", file=sys.stderr)
        return 1
    rows = read(args.logfile)
    if not rows:
        print(f"no 'Transcribed:' lines in {args.logfile}", file=sys.stderr)
        return 1

    print(f"{len(rows)} transcripts in {args.logfile}\n")
    print("threshold   rejected   kept")
    for t in CANDIDATES:
        rejected = sum(1 for _, lp in rows if lp < t)
        print(f"{t:9.2f}   {rejected:8d}   {len(rows) - rejected:4d}")

    lo, hi = args.band
    band = sorted((r for r in rows if lo <= r[1] < hi), key=lambda r: -r[1])
    print(f"\n--- {len(band)} in [{lo}, {hi}) "
          f"-- read these: are they commands or salad? ---")
    for text, lp in band:
        print(f"{lp:7.2f}  {text[:90]!r}")

    below = sorted((r for r in rows if r[1] < lo), key=lambda r: -r[1])
    print(f"\n--- {len(below)} below {lo} (still rejected) ---")
    for text, lp in below:
        print(f"{lp:7.2f}  {text[:90]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
