#!/usr/bin/env python3
"""Record which zone of a room he is in. Log only -- nothing acts on it.

His instruction was "just log it first", so this is the instrument that
gathers the evidence before any behaviour is wired to it. It polls ONE
room radar read-only through ``jarvis.roomsensor`` -- the same module, the
same URL rule, the same circuit breaker -- fuses each reading through
``jarvis.zones`` and appends one JSONL line per COMMITTED zone change.

IT IS GOVERNED BY OFFLINE MODE, and that is not a decoration: it builds the
same ``jarvis.sensing.SensingPolicy`` the app builds and hands it to the
sensor, so while sensing is denied no request leaves this process and
``reads`` stays at 0. The first pass claimed that and then constructed a
policy-less ``RoomSensor``, which meant an unattended ``--for 3600`` run
would have gone on polling the radar through a switch that was off. A
policy with no state file starts OFFLINE (the fail-safe), so an unexpected
"radar sensing is DENIED" line at start-up means the switch, not a bug.

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

IT REFUSES RATHER THAN GUESSES. ``jarvis.zones.read_zones`` validates the
whole ``zones`` section in one pass against a declared shape, and ANY key
in it that is the wrong shape -- the section, the enabled flag, the dwell,
a log key, the rooms list, a room entry, a room field, a band entry or a
band field -- means this exits 2 and names the dotted config path, printing
nothing but the reason. Only a key that is entirely ABSENT falls back to a
default, and the built-in office ladder answers for one case only: a config
with no ``zones.rooms`` key at all.
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
from jarvis.sensing import SensingPolicy                         # noqa: E402

DEFAULT_POLL_S = 2.0            # presence.rooms_poll_s, the fabric's cadence


def build_sensor(cfg, url: str, policy_path=None) -> RoomSensor:
    """A RoomSensor under the SAME sensing policy the app puts it under.

    ``scripts/vision_selfcheck.py`` and ``scripts/gesture_selfcheck.py``
    build their policy exactly this way. Without it the offline switch
    simply does not reach this process -- ``RoomSensor.blocked`` answers ""
    when there is no policy -- and this instrument is meant to be left
    running for an hour.

    A policy that cannot be built at all is fatal here rather than
    ignored: an ungoverned poll loop is the thing being avoided, so the
    caller gets the exception instead of a sensor that quietly ignores the
    switch. ``policy_path`` is the state-file seam the tests drive.
    """
    policy = SensingPolicy(cfg=cfg, path=policy_path)
    return RoomSensor(url, policy=policy)


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
    # ONE validation pass for the whole zones section, and every value
    # below comes off it. Asking jarvis.zones one question at a time would
    # walk the config once per question and print each complaint that many
    # times.
    zones = zn.read_zones(cfg)
    zmap = zones.rooms.get(room)
    if zmap is None and zones.bridge and room == "office":
        zmap = zn.ZoneMap.office()
    if zmap is None:
        # LOUD, and it records nothing. A config that got the zones section
        # wrong is refused by name rather than replaced by the built-in
        # ladder, because a log written against the bands he thought he had
        # replaced looks like it worked. See jarvis/zones.py: read_zones.
        print("no zones for %r -- NOTHING will be recorded." % room,
              file=sys.stderr)
        why = zones.why(room)
        if why and why not in zones.refused.values():
            # The room-level reason, when it is not already one of the
            # dotted paths printed below.
            print("  %s" % why, file=sys.stderr)
        print("  usable rooms: %s"
              % (", ".join(sorted(zones.rooms)) or "(none)"), file=sys.stderr)
        for key in sorted(zones.refused):
            # Keyed by the dotted config path, because that is the line he
            # has to go and edit.
            print("  refused: %s" % zones.refused[key], file=sys.stderr)
        return 2
    for key in sorted(zones.refused):
        print("WARNING: %s" % zones.refused[key], file=sys.stderr)

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

    sensor = build_sensor(cfg, url)
    if not sensor.configured:
        print("%r is not an http(s) URL." % url, file=sys.stderr)
        return 2
    if sensor.blocked:
        # Not fatal: offline mode can be lifted mid-run and the loop picks
        # it up. But it must be said, or an hour of "no opinion" reads as
        # a dead radar rather than a switch that is off.
        print("NOTE: radar sensing is DENIED right now (%s); no request will "
              "leave this process until that changes." % sensor.blocked,
              file=sys.stderr)
    dwell = args.dwell if args.dwell is not None else zones.dwell_s
    log_file = zn.ZoneLog(Path(args.log) if args.log else zones.log_path,
                          max_bytes=zones.log_max_bytes,
                          keep=zones.log_keep)
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
