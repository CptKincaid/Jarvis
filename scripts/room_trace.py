#!/usr/bin/env python
"""What his radars actually see, as numbers, while he walks a known route.

WHY THIS EXISTS. Every room sensor Jarvis reads answers ONE BIT -- "somebody
is here" -- and on 2026-09-11 both rooms answered "a still target at about
three metres" AT THE SAME TIME. He can only be in one of them, so at least
one of those bits is wrong, and the vote has been running on it all week.

The sensors already publish far more than the bit: moving distance, still
distance and detection distance, each throttled to ONE SECOND, plus the
moving and still target bits separately. `roomfabric` reads none of it --
the device YAML says so in as many words: "These three are for YOUR eyes on
the device's own web page while you decide where to mount it -- Jarvis never
reads them."

So before anyone adds a reed switch to a door, this asks whether the answer
is already on the wire. Distance is a DIRECTION: the kitchen unit faces the
front door, so somebody arriving appears at the far end of the beam and
comes closer, and somebody leaving recedes to the far end and vanishes.
A through-the-wall ghost, which is what the mount note warns about ("the
LD2410 reads through plasterboard and will hold presence ON from the
corridor"), should sit at a distance that does not move like a person.

NOTHING HERE TOUCHES A CAMERA, A MICROPHONE OR ANY OF HIS DATA. It reads
two numbers off two sensors on his own LAN and writes them to a CSV he owns.
It does not speak, does not touch the running Jarvis, and changes nothing.

HOW HE USES IT
    ~/vss_env/bin/python scripts/room_trace.py            # until Ctrl-C
    ~/vss_env/bin/python scripts/room_trace.py --seconds 300

While it runs, type a label and press enter whenever he starts doing
something, so the trace is annotated with the truth:

    at the desk        sitting still in the office
    to the kitchen     walking there
    out                going out of the front door
    back               coming back in
    bedroom            in the bedroom

The more of those, the sharper the thresholds. Five minutes with a couple
of trips out and back is enough to answer three questions that are
currently guesses: what distance the door crossing sits at, whether a
through-wall ghost can be told from a person by distance alone, and
whether the bedroom is visible at all.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOMS = {"kitchen": "192.168.50.52", "office": "192.168.50.51"}
ENTITIES = (("binary_sensor", "Presence", "presence"),
            ("binary_sensor", "Moving target", "moving"),
            ("binary_sensor", "Still target", "still"),
            ("sensor", "Moving distance", "moving_cm"),
            ("sensor", "Still distance", "still_cm"),
            ("sensor", "Detection distance", "detect_cm"))
TIMEOUT_S = 2.0
COLUMNS = ["t", "clock", "label"] + [f"{room}_{key}"
                                     for room in ROOMS
                                     for _d, _n, key in ENTITIES]


def read_one(ip: str, domain: str, name: str):
    """One entity's value, or None. Never raises: a sensor that does not
    answer is a hole in the trace, not the end of it."""
    url = f"http://{ip}/{domain}/{urllib.parse.quote(name)}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as res:
            return json.loads(res.read().decode("utf-8")).get("value")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def sample() -> dict:
    row = {}
    for room, ip in ROOMS.items():
        for domain, name, key in ENTITIES:
            row[f"{room}_{key}"] = read_one(ip, domain, name)
    return row


def _label_reader(box: list) -> None:
    """His typed labels, on their own thread so the sampling never waits."""
    for line in sys.stdin:
        text = line.strip()
        if text:
            box[0] = text
            print(f"    [{datetime.now():%H:%M:%S}] label -> {text!r}",
                  file=sys.stderr, flush=True)


def summarise(rows: list) -> None:
    if not rows:
        print("no samples")
        return
    print("\n---- what the two rooms said -------------------------------")
    for room in ROOMS:
        vals = [r[f"{room}_still_cm"] for r in rows
                if isinstance(r.get(f"{room}_still_cm"), (int, float))]
        on = sum(1 for r in rows if r.get(f"{room}_presence") is True)
        mov = sum(1 for r in rows if r.get(f"{room}_moving") is True)
        print(f"  {room:8} presence ON {on:4}/{len(rows)}   moving {mov:4}"
              f"   still_cm n={len(vals)}"
              + (f" min {min(vals)} median {sorted(vals)[len(vals)//2]} max {max(vals)}"
                 if vals else ""))
    both = sum(1 for r in rows
               if r.get("kitchen_presence") is True and r.get("office_presence") is True)
    print(f"\n  BOTH rooms occupied at once: {both} of {len(rows)} samples"
          f" ({100.0 * both / len(rows):.0f}%)")
    print("  He can only be in one, so that percentage is the size of the"
          " problem the bit alone cannot solve.")
    labels = [r["label"] for r in rows if r.get("label")]
    if labels:
        print("\n  labelled stretches:", ", ".join(sorted(set(labels))))
    else:
        print("\n  NO LABELS TYPED -- the trace records what the sensors said"
              "\n  but not what was true, so it can set no thresholds.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long (default: until Ctrl-C)")
    ap.add_argument("--every", type=float, default=1.0,
                    help="seconds between samples (the sensors throttle at 1)")
    ap.add_argument("--out", default=None, help="CSV path")
    args = ap.parse_args()

    out = Path(args.out or (Path.home() /
               f"room-trace-{datetime.now():%Y%m%d-%H%M}.csv"))
    print(__doc__.split("HOW HE USES IT")[0].strip()[:0] or "", end="")
    print(f"writing {out}")
    print("type a label + enter whenever you change what you are doing"
          " (at the desk / to the kitchen / out / back / bedroom); Ctrl-C to stop\n")

    box = ["start"]
    threading.Thread(target=_label_reader, args=(box,), daemon=True).start()

    rows, t0 = [], time.monotonic()
    try:
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            while args.seconds is None or time.monotonic() - t0 < args.seconds:
                row = sample()
                row["t"] = round(time.monotonic() - t0, 1)
                row["clock"] = datetime.now().strftime("%H:%M:%S")
                row["label"] = box[0]
                w.writerow(row)
                fh.flush()
                rows.append(row)
                k = row.get("kitchen_still_cm")
                o = row.get("office_still_cm")
                print(f"  {row['clock']}  kitchen {str(row.get('kitchen_presence')):5}"
                      f" {str(k):>5}cm   office {str(row.get('office_presence')):5}"
                      f" {str(o):>5}cm   [{row['label']}]", flush=True)
                time.sleep(max(0.2, args.every))
    except KeyboardInterrupt:
        print("\nstopped")
    summarise(rows)
    print(f"\nCSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
