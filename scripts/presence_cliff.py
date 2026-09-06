#!/usr/bin/env python
"""The 2026-09-05 cliff, MEASURED end to end and printed as a CURVE.

    ~/vss_env/bin/python scripts/presence_cliff.py
    ~/vss_env/bin/python scripts/presence_cliff.py --recency-min 30
    ~/vss_env/bin/python scripts/presence_cliff.py --ever     # the first cut

NUMBERS ONLY. Nothing here touches the network, a sensor, the live app,
the camera or the microphone: the rooms are stubs, the phone is a flag,
the mic leg is arithmetic on a fake clock, and the stuck-room state file
is a temp file. It drives the REAL PresenceSentinel, ThreeLegProbe,
StuckRooms and DoorWatch on the 60 s home poll the live box runs.

SCENARIO A -- THE LATCH (defect 1, and the night itself). He is at his
desk for --warm-min with his phone on the Wi-Fi, so the office run is
stamped every minute: the realistic latch, the one the first replay test
did not have. He leaves for T minutes. The phone drops, the office radar
never clears, the camera cannot look, and the mic last heard him as he
walked out (the worst case for the greeting, on purpose). On his return
the kitchen lights. GREETED means the sentinel read "away" at that moment,
which is the one thing door_arrival needs. FLIP is the minute of the trip
at which the sentinel reached away; FAULT says whether the 45-minute
stuck-room detector had already dropped the office by then.

SCENARIO B -- THE DESK (defect 2, the false away). He never leaves. His
phone naps for good at minute 0, the office radar is correctly on, the
camera is dark. FLIP is the minute the sentinel falsely called him away,
or "never". With a spoken turn every N minutes it must be never.

--ever reproduces the first cut's tie-break ("has anything EVER agreed
with this run?"): a run stamped at least once is home for as long as it
lasts, a run never stamped is away on the spot, and the voter has no mic.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_TMP = tempfile.mkdtemp(prefix="presence-cliff-")
os.environ.setdefault("JARVIS_LOG_DIR", _TMP)        # never /tmp/vss_voice
os.environ.setdefault("JARVIS_MEMORY_DIR", _TMP)     # never his roomruns.json

from jarvis import arrival, presence, roomfabric, stuckroom  # noqa: E402


class Reader:
    configured, blocked, paused = True, "", False

    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value

    def status(self):
        return {}


class Cfg:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class Box:
    """One fake flat: two radars, a phone flag, a turn ledger, a clock."""

    def __init__(self, *, recency_s, mic_window_s, mic: bool, ever: bool):
        self.clock = {"t": 3_000_000.0}
        now = lambda: self.clock["t"]                       # noqa: E731
        self.office, self.kitchen = Reader(True), Reader(False)
        self.fab = roomfabric.from_readers(
            [("office", self.office), ("kitchen", self.kitchen)],
            now=now, enter_hold_s=0.0, poll_s=2.0)
        self.fab.tick()
        self.stuck = stuckroom.StuckRooms(
            path=Path(_TMP) / ("run-%s.json" % uuid.uuid4().hex),
            publish=lambda e: None, now=now)
        self.phone = {"up": True}
        self.last_turn = {"t": None}

        def mic_reader():
            t = self.last_turn["t"]
            return None if t is None else now() - t

        self.leg = presence.ThreeLegProbe(
            fabric=self.fab, stuck=self.stuck,
            phone=lambda ip, mac: self.phone["up"],
            mic=(mic_reader if mic else None),
            recency_s=recency_s, mic_window_s=mic_window_s)
        if ever:
            # THE FIRST CUT'S QUESTION, byte for byte: "has anything EVER
            # agreed with this run?" -- an unstamped run was away on the
            # spot, a stamped one home for as long as the run lasted.
            self.leg.recency_s = float("inf")

            def ever_agreed(stuck=self.stuck):
                rows = [r for r in stuck.status().values() if r.get("run_s")]
                if not rows:
                    return None
                return (0.0 if any(r.get("corroborated_s_ago") is not None
                                   for r in rows) else float("inf"))
            self.leg._agreed_s_ago = ever_agreed
        self.s = presence.PresenceSentinel(
            Cfg(**{"presence.phone_ip": "192.0.2.34", "presence.enabled": True}),
            publish=lambda e: None, probe_fn=self.leg, now=now, poll_s=60.0)

    def minute(self):
        self.clock["t"] += 60.0
        self.fab.tick()
        self.s.tick()

    def speak(self):
        self.last_turn["t"] = self.clock["t"]
        self.stuck.corroborate("mic")          # app._on_wake does the same


def latch(trip_min, *, warm_min, spoke_at_door, **kw):
    box = Box(**kw)
    for _ in range(warm_min):
        box.minute()
    if spoke_at_door:
        box.speak()
    box.phone["up"] = False                     # he is out; the office stays ON
    flip = None
    for m in range(1, trip_min + 1):
        box.minute()
        if flip is None and box.s.state == "away":
            flip = m
    greeted = arrival.DoorWatch().observe(
        room="kitchen", away=(box.s.home is False), state=box.s.state) is True
    return flip, greeted, ("office" in box.stuck.faulted())


def desk(*, warm_min, turn_every_min, limit, **kw):
    box = Box(**kw)
    for _ in range(warm_min):
        box.minute()
    box.phone["up"] = False                     # the nap begins; he stays put
    for m in range(1, limit + 1):
        if turn_every_min and m % turn_every_min == 0:
            box.speak()
        box.minute()
        if box.s.state == "away":
            return m
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recency-min", type=float, default=15.0)
    ap.add_argument("--mic-window-min", type=float, default=10.0)
    ap.add_argument("--warm-min", type=int, default=60)
    ap.add_argument("--away-min", type=int, default=12,
                    help="presence.away_after_min; 12 on his box")
    ap.add_argument("--trips", default="5,10,15,20,25,26,27,28,30,35,40,45,"
                                       "50,55,56,57,58,60,75,90,100,120")
    ap.add_argument("--ever", action="store_true",
                    help="the first cut: infinite window, no mic")
    ap.add_argument("--no-mic", action="store_true",
                    help="the voter has no mic reader (MIC_UNKNOWN)")
    a = ap.parse_args(argv)
    presence.DEFAULT_AWAY_MIN = a.away_min
    kw = dict(recency_s=a.recency_min * 60.0, mic_window_s=a.mic_window_min * 60.0,
              mic=not (a.ever or a.no_mic), ever=a.ever)
    label = ("FIRST CUT: ever-corroborated, no mic" if a.ever else
             "recency %g min, mic window %g min%s" % (
                 a.recency_min, a.mic_window_min,
                 " (no mic reader)" if a.no_mic else ""))
    print("== %s; away grace %d min; %d min at the desk first ==" % (
        label, a.away_min, a.warm_min))
    print("\nA. THE LATCH -- he leaves for T minutes; office ON throughout")
    print("   trip  flip  greeted  office-faulted")
    for t in (int(x) for x in a.trips.split(",")):
        flip, greeted, faulted = latch(t, warm_min=a.warm_min, spoke_at_door=True, **kw)
        print("   %4d  %4s  %-7s  %s" % (t, "-" if flip is None else flip,
                                       "YES" if greeted else "no",
                                       "yes" if faulted else "no"))
    print("\nB. THE DESK -- he never leaves; phone naps at minute 0")
    print("   (warm 0: the run had never been stamped -- a restart, or a nap "
          "that began before the radar saw him)")
    for warm in (0, 30):
        for every in (None, 30, 20, 10, 5):
            flip = desk(warm_min=warm, turn_every_min=every, limit=180, **kw)
            print("   warm %2d  turn every %-5s  false away at minute %s" % (
                warm, "never" if every is None else "%d" % every,
                "never (180 min)" if flip is None else flip))
    return 0


if __name__ == "__main__":
    sys.exit(main())
