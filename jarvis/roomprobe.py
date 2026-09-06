"""The instrument HE runs. Numbers only, and read-only.

    ~/vss_env/bin/python -m jarvis.roomprobe --seconds 120

WHAT IT ANSWERS, per room, over a bounded window:

  * is this room reporting occupancy that NEVER CLEARS -- the 2026-09-05
    failure, where the office read ON for 30 of 30 samples with the flat
    empty while the kitchen was correctly OFF
  * what is the target DISTANCE -- moving, still and detection, as median,
    min and max, so a still body sitting at the edge of the configured
    band is visible as a number rather than a hunch
  * is it MOVING or STILL
  * does that match the CONFIGURED WINDOW -- his room profile's near and
    far limits, so "he sits outside the range he set" is a fact rather
    than an inference
  * and the health of the leg itself: HTTP errors, latency, Wi-Fi, uptime

WHAT IT NEVER DOES, and both of these are hard rules rather than
preferences:

  * IT NEVER OPENS A CAMERA OR A MICROPHONE. There is no image and no
    audio anywhere in this module, and no field of the report can carry
    one (tests/test_roomprobe.py pins the field names).
  * IT NEVER WRITES TO A SENSOR. No OTA, no config push, no reboot, no
    gate or timeout change. Every request is a GET at an entity URL. If
    the fix turns out to be a setting on the device, this prints the
    INSTRUCTION and he makes the change himself.

Polling is floored at one sample per 2 s and the window is hard-capped, so
the instrument cannot become a load on his flat's hardware.

THE URL RULE, MEASURED AND EXPENSIVE. ESPHome's web_server v2 serves an
entity at its NAME, not its object_id: ``/sensor/Still%20distance``, not
``/sensor/still_distance``. Every guessed object_id URL was a 404 once and
presence failed SILENTLY for it. ``entity_url`` is the one place that
knows, and a test pins it.

WHAT HIS FIRMWARE DOES NOT EXPOSE, measured 2026-09-05: there are no
per-gate ENERGY sensors on the flashed image (``sensor/Moving energy`` and
``sensor/Still energy`` both 404). Energy is what would separate "a body"
from "a curtain" definitively, so its absence is worth knowing; the report
says so rather than printing a blank, and names the YAML that would add it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

# One sample per 2 s at the very fastest: his ESP32s are live hardware in
# his flat and an instrument must not become a load on them.
MIN_INTERVAL_S = 2.0
DEFAULT_INTERVAL_S = 2.0
DEFAULT_SECONDS = 120.0
# Every loop hard-bounded. Ten minutes is long enough to catch a latch
# that a 60 s window would miss and short enough that he waits for it.
MAX_SECONDS = 600.0
HTTP_TIMEOUT_S = 3.0

# His two rooms, from jarvis-office.yaml / jarvis-kitchen.yaml.
ROOMS = {"office": "192.168.50.51", "kitchen": "192.168.50.52"}

# The entities that RESOLVE on the flashed firmware (measured 2026-09-05).
ENTITIES = (
    ("binary_sensor", "Presence"),
    ("binary_sensor", "Moving target"),
    ("binary_sensor", "Still target"),
    ("sensor", "Moving distance"),
    ("sensor", "Still distance"),
    ("sensor", "Detection distance"),
)
SETTINGS = (
    ("number", "Absence delay"),
    ("number", "Max move gate"),
    ("number", "Max still gate"),
    ("sensor", "WiFi signal"),
    ("sensor", "Uptime"),
)


def clamp_interval(value) -> float:
    try:
        return max(MIN_INTERVAL_S, float(value))
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S


def clamp_seconds(value) -> float:
    try:
        return max(MIN_INTERVAL_S, min(MAX_SECONDS, float(value)))
    except (TypeError, ValueError):
        return DEFAULT_SECONDS


def entity_url(base: str, domain: str, name: str) -> str:
    """``/<domain>/<NAME>``, percent-encoded. See the module docstring."""
    return "%s/%s/%s" % (base.rstrip("/"), domain, urllib.parse.quote(name))


def read_entity(base: str, domain: str, name: str, opener=None,
                timeout: float = HTTP_TIMEOUT_S):
    """One GET. Returns the decoded JSON dict, or None. Never writes."""
    opener = opener or urllib.request.urlopen
    with opener(entity_url(base, domain, name), timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


@dataclass
class Sample:
    """One sweep of one room. Numbers and booleans only."""

    at: float = 0.0
    presence: Optional[bool] = None
    moving_target: Optional[bool] = None
    still_target: Optional[bool] = None
    moving_cm: Optional[float] = None
    still_cm: Optional[float] = None
    detect_cm: Optional[float] = None
    ms: Optional[float] = None
    error: str = ""


def _stat(values) -> Optional[dict]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return {"median": round(statistics.median(vals), 1),
            "min": round(min(vals), 1), "max": round(max(vals), 1),
            "sd": round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def summarise(room: str, samples, profile: Optional[dict] = None) -> dict:
    """The report, as numbers. Pure -- no network, so it is testable."""
    good = [s for s in samples if not s.error and s.presence is not None]
    errors = sum(1 for s in samples if s.error)
    lat = [s.ms for s in samples if s.ms is not None]
    on = sum(1 for s in good if s.presence)

    window_s = 0.0
    if len(samples) > 1:
        window_s = round(float(samples[-1].at) - float(samples[0].at), 1)

    report = {
        "room": room,
        "samples": len(good),
        "errors": errors,
        "window_s": window_s,
        "median_ms": round(statistics.median(lat), 1) if lat else None,
        "on": on,
        # NEVER CLEARS is the 2026-09-05 signature, and it is a claim about
        # THIS WINDOW only -- which is why window_s sits beside it.
        "never_clears": (bool(good) and on == len(good)) if good else None,
        "on_fraction": (round(on / len(good), 3) if good else None),
        "moving_on": sum(1 for s in good if s.moving_target),
        "still_on": sum(1 for s in good if s.still_target),
        "moving_cm": _stat(s.moving_cm for s in samples),
        "still_cm": _stat(s.still_cm for s in samples),
        "detect_cm": _stat(s.detect_cm for s in samples),
        "window_m": None,
        "outside_window": None,
    }

    # Against the window HE configured, when the profile is known. This is
    # his own setting, so it needs no gate arithmetic and cannot be wrong
    # about the hardware: a body measured outside the band he asked to
    # watch is held only by whatever margin the gate above it happens to
    # give, and that is a one-line fix he can make himself.
    if profile:
        try:
            near = float(profile.get("nearest_m"))
            far = float(profile.get("range_m"))
        except (TypeError, ValueError):
            near = far = None
        if near is not None and far is not None:
            report["window_m"] = (near, far)
            seen = [v for s in samples
                    for v in (s.still_cm, s.moving_cm) if v is not None]
            if seen:
                report["outside_window"] = any(
                    v < near * 100.0 or v > far * 100.0 for v in seen)
    return report


def render(report: dict) -> str:
    """The report as text he can read. One room."""
    out = ["", "=" * 66,
           "ROOM %s -- %d samples over %ss, %d HTTP errors, %s ms median"
           % (report["room"], report["samples"], report["window_s"],
              report["errors"],
              report["median_ms"] if report["median_ms"] is not None else "-")]
    if report["never_clears"] is None:
        out.append("  no usable readings -- nothing can be said about this room")
        return "\n".join(out)

    if report["never_clears"]:
        out.append("  OCCUPANCY: NEVER CLEARED. %d of %d samples read ON, with"
                   % (report["on"], report["samples"]))
        out.append("    no break at all across the whole %ss window."
                   % report["window_s"])
        out.append("    If the room was EMPTY for that window, this is the")
        out.append("    2026-09-05 fault: a latched radar. If you were in it,")
        out.append("    this is simply correct -- the instrument cannot tell,")
        out.append("    and only you know which it was.")
    else:
        out.append("  OCCUPANCY: %d of %d samples ON (%.0f%%) -- it clears."
                   % (report["on"], report["samples"],
                      100.0 * (report["on_fraction"] or 0.0)))
    out.append("  moving target ON %d, still target ON %d"
               % (report["moving_on"], report["still_on"]))

    for key, label in (("moving_cm", "moving  "), ("still_cm", "still   "),
                       ("detect_cm", "detected")):
        st = report.get(key)
        if st:
            out.append("  %s distance  median %s cm  (min %s, max %s, sd %s, "
                       "n=%d)" % (label, st["median"], st["min"], st["max"],
                                  st["sd"], st["n"]))
    if report.get("window_m"):
        near, far = report["window_m"]
        out.append("  configured watch window: %.2f m to %.2f m" % (near, far))
        if report["outside_window"]:
            out.append("    ** A TARGET WAS MEASURED OUTSIDE THAT WINDOW. **")
            out.append("    The body is being held by the gate above your")
            out.append("    setting rather than by the setting itself, so a")
            out.append("    still target can drop out of it without warning.")
            out.append("    FIX IT YOURSELF in the sensor setup sheet: raise")
            out.append("    the watch range past the measured distance above,")
            out.append("    then re-flash. I do not write to your sensors.")
        else:
            out.append("    every measured target sits inside it")
    return "\n".join(out)


def load_profile(room: str) -> Optional[dict]:
    """The geometry keys of his room profile, and ONLY those.

    ``~/.config/jarvis/room-sensors/<room>.json`` also holds his Wi-Fi PSK
    and an OTA password. This reads the two numbers it needs by name and
    nothing else ever leaves the function -- a previous session cat'd one
    of these files to get an ``ip`` and printed his Wi-Fi password into a
    transcript.
    """
    try:
        from pathlib import Path
        path = Path.home() / ".config" / "jarvis" / "room-sensors" / ("%s.json" % room)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {"nearest_m": raw.get("nearest_m"), "range_m": raw.get("range_m")}
    except Exception:  # noqa: BLE001 - a missing profile costs the window check
        return None


def sweep(base: str, opener=None) -> Sample:
    """One read of every live entity in one room. Read-only."""
    t0 = time.time()
    s = Sample(at=t0)
    try:
        for domain, name in ENTITIES:
            data = read_entity(base, domain, name, opener=opener)
            value = data.get("value")
            if name == "Presence":
                s.presence = bool(value)
            elif name == "Moving target":
                s.moving_target = bool(value)
            elif name == "Still target":
                s.still_target = bool(value)
            elif name == "Moving distance":
                s.moving_cm = float(value)
            elif name == "Still distance":
                s.still_cm = float(value)
            elif name == "Detection distance":
                s.detect_cm = float(value)
    except Exception as exc:  # noqa: BLE001 - a dead room is a number too
        s.error = type(exc).__name__
    s.ms = round((time.time() - t0) * 1000.0, 1)
    return s


def settings(base: str, opener=None) -> dict:
    """The device's own configuration, read back. Never written."""
    out = {}
    for domain, name in SETTINGS:
        try:
            out[name] = read_entity(base, domain, name, opener=opener).get("state")
        except Exception as exc:  # noqa: BLE001
            out[name] = "unreadable (%s)" % type(exc).__name__
    return out


def run(rooms=None, seconds: float = DEFAULT_SECONDS,
        interval: float = DEFAULT_INTERVAL_S, opener=None,
        sleep=time.sleep, out=print) -> dict:
    """Poll for a BOUNDED window and print a report per room."""
    rooms = dict(rooms or ROOMS)
    seconds, interval = clamp_seconds(seconds), clamp_interval(interval)
    # Hard-bounded twice: by sample count AND by wall clock, so neither a
    # slow network nor a clock change can make this run away.
    budget = int(seconds / interval) + 1
    deadline = time.time() + seconds + interval

    out("polling %s for %.0fs, one sample every %.0fs (read-only)"
        % (", ".join(rooms), seconds, interval))
    for name, ip in rooms.items():
        out("  %-8s %s" % (name, ip))
    out("")
    for name, ip in rooms.items():
        cfg = settings("http://%s" % ip, opener=opener)
        out("%s settings: %s" % (name, ", ".join(
            "%s=%s" % (k, v) for k, v in cfg.items())))
    out("\n(no per-gate ENERGY sensors on this firmware -- `sensor/Moving "
        "energy`\n and `sensor/Still energy` both 404. To add them, put the "
        "ld2410 per-gate\n sensors into the room YAML and re-flash. That is "
        "your change to make.)\n")

    collected = {name: [] for name in rooms}
    for i in range(budget):
        if time.time() > deadline:
            break
        for name, ip in rooms.items():
            collected[name].append(sweep("http://%s" % ip, opener=opener))
        if i < budget - 1:
            sleep(interval)

    reports = {}
    for name in rooms:
        reports[name] = summarise(name, collected[name], load_profile(name))
        out(render(reports[name]))
    out("")
    return reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Read Hunter's room sensors and report NUMBERS. "
                    "Read-only: it never writes to a sensor.")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                    help="window length (capped at %d)" % MAX_SECONDS)
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                    help="seconds between samples (floor %.0f)" % MIN_INTERVAL_S)
    ap.add_argument("--room", action="append", default=None,
                    help="office | kitchen (default: both)")
    args = ap.parse_args(argv)
    rooms = ({r: ROOMS[r] for r in args.room if r in ROOMS}
             if args.room else dict(ROOMS))
    if not rooms:
        print("no known room named; try --room office or --room kitchen")
        return 2
    run(rooms=rooms, seconds=args.seconds, interval=args.interval)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
