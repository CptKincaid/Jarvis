#!/usr/bin/env python3
"""Record which zone of a room he is in. Log only -- nothing acts on it.

His instruction was "just log it first", so this is the instrument that
gathers the evidence before any behaviour is wired to it. It polls ONE
room radar read-only through ``jarvis.roomsensor`` -- the same module, the
same URL rule, the same circuit breaker and the same offline-mode policy
Jarvis itself uses -- fuses each reading through ``jarvis.zones`` and
appends one JSONL line per COMMITTED zone change.

    scripts/zone_log.py --describe              # the ladder, no network
    scripts/zone_log.py --room office           # poll until Ctrl-C
    scripts/zone_log.py --room office --for 600 # ten minutes, then stop
    scripts/zone_log.py --room office --dry-run # print, write nothing

WHAT IT DOES NOT DO, on purpose:

* it never writes to the radar (no tuning, no power switch): read-only GETs
  and nothing else, so it cannot disturb a device Jarvis is also reading;
* it never opens a camera. The camera OVERRULES the radar in the zone
  model, so a run of this alone will never report the camera zone and the
  desk will read as "not in the room" while he sits in it. That is the
  expected shape of a radar-only record, not a fault -- see the module
  docstring of jarvis/zones.py for why the chair is where the radar is
  blind;
* it never starts Jarvis, and it is safe to run while Jarvis is running:
  two readers of one ESPHome endpoint is two HTTP GETs.

The URL comes from ``presence.room_sensor_url`` (or the matching entry in
``presence.rooms``) unless ``--url`` overrides it; the bands, the dwell and
the log path come from the ``zones`` section. A restart of Jarvis is not
needed for either -- this process reads the config itself.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import zones as zn                                   # noqa: E402
from jarvis.assistant_config import AssistantConfig              # noqa: E402
from jarvis.roomsensor import RoomSensor                         # noqa: E402

DEFAULT_POLL_S = 2.0            # presence.rooms_poll_s, the fabric's cadence


def room_url(cfg, room: str) -> str:
    """The configured radar URL for one room, "" when there is not one.

    Reads the plural ``presence.rooms`` first and falls back to the
    singular ``presence.room_sensor_url``, exactly as roomfabric does.
    """
    entries = cfg.get("presence.rooms", None)
    if isinstance(entries, (list, tuple)):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if " ".join(str(entry.get("name") or "").split()).lower() == room:
                return str(entry.get("url") or "").strip()
    return str(cfg.get("presence.room_sensor_url", "") or "").strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--room", default="office", help="which room's ladder")
    ap.add_argument("--url", default="", help="override the radar URL")
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL_S,
                    help="seconds between readings (default %.1f)" % DEFAULT_POLL_S)
    ap.add_argument("--dwell", type=float, default=None,
                    help="anti-chatter hold; default is zones.dwell_s")
    ap.add_argument("--for", dest="seconds", type=float, default=0.0,
                    help="stop after this many seconds (0 = until Ctrl-C)")
    ap.add_argument("--log", default="", help="override the JSONL path")
    ap.add_argument("--dry-run", action="store_true",
                    help="print transitions, write no file")
    ap.add_argument("--describe", action="store_true",
                    help="print the zone ladder and exit")
    args = ap.parse_args(argv)

    cfg = AssistantConfig.load()
    room = " ".join(str(args.room).split()).lower()
    zmap = zn.zone_map_for(cfg, room)
    if zmap is None:
        print("no zones for %r. zones.enabled is %r; configured rooms: %s"
              % (room, cfg.get("zones.enabled", True),
                 ", ".join(sorted(zn.zone_maps(cfg))) or "(none)"),
              file=sys.stderr)
        return 2

    print("%s -- the ladder:" % room)
    print(zmap.describe())
    gaps = ", ".join("%.2f-%s m" % (lo, "inf" if hi is None else "%.2f" % hi)
                     for lo, hi in zmap.gaps())
    print("  gaps (-> %s): %s" % (zn.UNPLACED, gaps))
    if args.describe:
        return 0

    url = args.url.strip() or room_url(cfg, room)
    if not url:
        print("no radar URL for %r: set presence.room_sensor_url (or the "
              "matching presence.rooms entry), or pass --url." % room,
              file=sys.stderr)
        return 2

    sensor = RoomSensor(url)
    if not sensor.configured:
        print("%r is not an http(s) URL." % url, file=sys.stderr)
        return 2
    dwell = args.dwell if args.dwell is not None else zn.dwell_s(cfg)
    log_file = zn.ZoneLog(Path(args.log) if args.log else zn.log_path(cfg),
                          max_bytes=zn.log_max_bytes(cfg),
                          keep=zn.log_keep(cfg))
    watcher = zn.ZoneWatcher(room, sensor, zmap, dwell_s=dwell,
                             log_file=log_file)
    if args.dry_run:
        # The tracker is what appends; unhooking it there leaves the counters
        # below meaningful (0 writes, 0 failures) rather than absent.
        watcher.tracker.log = None

    print("\npolling %s every %.1f s, dwell %.1f s -> %s"
          % (sensor.url, args.poll, dwell,
             "(dry run: nothing written)" if args.dry_run else log_file.path))
    print("the camera is NOT read here, so the desk will read as %r." % zn.ABSENT)
    started, polls = time.monotonic(), 0
    try:
        while True:
            change = watcher.poll()
            polls += 1
            if change is not None:
                rec = change.as_record()
                print("%s  %-22s -> %-22s  %s  %s"
                      % (rec["iso"], rec["old"], rec["new"], rec["rule"],
                         "-" if rec["distance_m"] is None
                         else "%.2f m" % rec["distance_m"]))
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            time.sleep(max(0.1, args.poll))
    except KeyboardInterrupt:
        print()
    print("%d polls, %d requests, %d transitions written, %d log failures"
          % (polls, sensor.reads, log_file.writes, log_file.failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
