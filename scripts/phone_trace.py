#!/usr/bin/env python3
"""What his phone's ARP row actually does -- INCLUDING while he is out.

WHY THIS EXISTS. Measured on his box 2026-09-11, 30 samples 15 s apart
while he was home:

    single `ping -c1 -W1` answered          7 / 30   (23%)
    kernel ARP resolution answered         15 / 15   (100%)

Every DELAY row this instrument created had become REACHABLE by the next
sample. His iPhone ignores unsolicited ICMP for minutes at a time and
answers ARP every time it is asked -- ARP is handled by the radio, ICMP
by an OS that is asleep. The presence probe asks with ICMP, so it
concluded "he left the flat" eleven polls in a row while he sat in the
office, and that is where the seven false departures of 09-10/11 came
from.

WHAT IS STILL UNMEASURED, and it is the reason this is a script he runs
rather than a patch I ship: what the row looks like when he is GENUINELY
OUT. Rewriting the probe to believe ARP over ICMP is only safe if a
departure still produces FAILED (or a vanished row) within a reasonable
time. If it does not, presence would never go away again, which is the
same bug pointing the other way.

HOW TO RUN IT. Start it, then leave with your phone for at least fifteen
minutes and come back:

    ~/vss_env/bin/python scripts/phone_trace.py --minutes 60

It prints one line per sample and writes a CSV beside itself. It reads
the address through AssistantConfig and never prints it. It sends ICMP
echo to one address on your own LAN -- the same packet Jarvis already
sends every sixty seconds -- and stores nothing else.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.assistant_config import AssistantConfig  # noqa: E402

# How long to let the kernel finish resolving before re-reading the row.
# net.ipv4.neigh.default.delay_first_probe_time is 5 s, then unicast
# probes at 1 s; 8 s clears both on an unloaded box.
SETTLE_S = 8.0
PRESENT = ("REACHABLE", "PERMANENT")      # NOT delay/probe: those mean "asking"


def _run(argv, timeout):
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except Exception:                     # noqa: BLE001 - a probe, not a service
        return None


def arp_state(ip: str) -> str:
    res = _run(["ip", "-4", "neigh"], 3.0)
    if res is None:
        return "NO-IP-CMD"
    for line in (res.stdout or "").splitlines():
        f = line.split()
        if f and f[0] == ip:
            return f[-1]
    return "ABSENT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ip = (AssistantConfig.load().get("presence.phone_ip") or "").strip()
    if not ip:
        print("presence.phone_ip is not configured; nothing to trace.")
        return 1

    out = Path(args.out or (Path(__file__).resolve().parent.parent /
                            "phone_trace.csv"))
    deadline = time.monotonic() + args.minutes * 60.0
    counts: dict[str, int] = {}
    pings = [0, 0]                         # [hit, total]

    print(f"tracing for {args.minutes:.0f} min, one sample every "
          f"{args.every:.0f}s -> {out}")
    print("  time      before      ping   after")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "before", "ping", "after"])
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            before = arp_state(ip)
            res = _run(["ping", "-c", "1", "-W", "1", ip], 3.0)
            hit = res is not None and res.returncode == 0
            pings[1] += 1
            pings[0] += 1 if hit else 0
            time.sleep(SETTLE_S)
            after = arp_state(ip)
            counts[after] = counts.get(after, 0) + 1
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"  {stamp}  {before:10s}  {'HIT ' if hit else 'MISS'}  "
                  f"{after}", flush=True)
            w.writerow([stamp, before, "hit" if hit else "miss", after])
            fh.flush()
            time.sleep(max(0.0, args.every - (time.monotonic() - t0)))

    print(f"\nsamples                 {pings[1]}")
    print(f"ICMP answered           {pings[0]}/{pings[1]}")
    print("settled ARP state after each ping:")
    for state, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        mark = "  <- reads as PRESENT" if state in PRESENT else ""
        print(f"  {state:12s} {n}{mark}")
    print(f"\nCSV: {out}")
    print("Send me the summary. The line that decides the fix is whether a "
          "real absence ever produced FAILED or ABSENT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
