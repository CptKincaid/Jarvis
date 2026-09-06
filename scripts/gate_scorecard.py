#!/usr/bin/env python3
"""What the owner gate WOULD have done to you, in numbers you can read.

The gate ships in SHADOW: it works out a verdict for every voice turn and
then answers the turn anyway. This reads the verdicts back.

    ~/vss_env/bin/python scripts/gate_scorecard.py            # last 7 days
    ~/vss_env/bin/python scripts/gate_scorecard.py --days 1
    ~/vss_env/bin/python scripts/gate_scorecard.py --hours 6
    ~/vss_env/bin/python scripts/gate_scorecard.py --json     # for a plot

SAFE TO RUN WHILE JARVIS IS LIVE. The only file of yours it touches is the
one file it reads -- /tmp/vss_voice/gate.jsonl -- and it never writes it,
never speaks to the running process, and never opens the log, the people
book, the voiceprint or the gallery. Measured: /tmp/vss_voice is byte-for-
byte unchanged after a run, and a row half-written by the app is skipped.

IT IS NOT SILENT, THOUGH, AND THE HONEST VERSION OF THAT SENTENCE IS THIS:
importing jarvis.config appends a startup line to this repository's own
ad-hoc log under /tmp/jarvis-adhoc and probes the GPU, exactly as every
other script here does. Nothing of yours is in that and your live log
directory is untouched -- but a file whose whole value is honesty about
itself must not advertise itself as side-effect-free, because that claim was
measured on a real run and it was false.

WHAT IT CANNOT TELL YOU, said here and again in every printout: the ledger
records what the GATE DECIDED, never who was really speaking. No line in the
output is a measurement of accuracy, and there is no number in it that makes
switching to enforce safe.

WHAT IT SAYS WHEN IT CANNOT READ: a missing, unreadable, corrupt or
permission-denied ledger prints a sentence at the TOP saying so, never a
table of zeroes. A scorecard that silently reports 0 when it could not read
something is worse than no scorecard at all, because it gets acted on. See
jarvis/gateledger.py for the shape of a row and for what may never be
written into one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import gateledger as gl                        # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=None,
                    help="how far back to look (default 7)")
    ap.add_argument("--hours", type=float, default=None,
                    help="how far back to look, in hours")
    ap.add_argument("--path", default=str(gl.DEFAULT_PATH),
                    help="the ledger to read (default %s)" % gl.DEFAULT_PATH)
    ap.add_argument("--buckets", type=int, default=None,
                    help="how many rows the trend is split into")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="the same numbers as JSON, for plotting")
    args = ap.parse_args(argv)

    if args.hours is not None:
        window_s = max(60.0, float(args.hours) * 3600.0)
    elif args.days is not None:
        window_s = max(60.0, float(args.days) * gl.DAY_S)
    else:
        window_s = gl.DEFAULT_WINDOW_S

    now = time.time()
    # read_report, NOT read: the report carries what could not be read, and
    # that is the half the printout must not lose.
    card = gl.summarise(gl.read_report(args.path), now=now,
                        window_s=window_s, buckets=args.buckets)
    if args.as_json:
        out = card.as_dict()
        out["window_s"] = window_s
        out["now"] = now
        out["path"] = str(args.path)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    print(gl.render(card, now=now, window_s=window_s, path=str(args.path)))
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
