#!/usr/bin/env python3
"""Set up an ESP32 + LD2410 room sensor: where it is, what it watches, flash it.

One script per sensor, keyed by the ROOM it sits in. It owns a profile
(network, mount geometry, radar tuning), renders the ESPHome YAML from
``scripts/esphome/jarvis-room-sensor.yaml``, flashes it over USB the first
time and over Wi-Fi after that, sets the radar's runtime knobs from the
geometry, and checks the result through the same module Jarvis reads with.

    scripts/room_sensor.py doctor                 # what is missing, and the fix
    scripts/room_sensor.py init office --ssid X --password Y --ip 192.168.50.60 \
        --preset desk --range 3.5 --nearest 2.0
    scripts/room_sensor.py flash office           # USB first time, OTA after
    scripts/room_sensor.py tune office            # push the geometry to the radar
    scripts/room_sensor.py check office           # what Jarvis would read, now
    scripts/room_sensor.py watch office           # every transition, timed
    scripts/room_sensor.py install-config office  # write presence.* (backs up)

THE PART THAT DECIDES WHETHER THIS WORKS AT ALL IS THE MOUNT, NOT THE CODE.
Two numbers from Hi-Link's own datasheet, both load-bearing, neither of them
in ``docs/room-sensor.md``:

* **Nothing at all inside 0.75 m.** "Detection distance 0.75m ~ 6m" -- closer
  than that and the module is blind.
* **No STILL-target detection inside 1.5 m.** The serial-protocol table lists
  rest sensitivity for gates 0 and 1 as "-(not settable)": not a threshold of
  zero, not implemented. Inside 1.5 m the LD2410 is a motion sensor -- a PIR,
  which is the thing this whole feature exists to avoid, because a man sitting
  still at a desk stops existing.

So ``init`` REFUSES a mount whose subject sits inside 1.5 m unless you pass
``--motion-only``, which is honest for a doorway and wrong for a desk. See
docs/room-sensor-bedroom.md for where the two numbers come from, gate by gate.

GATES ARE 0.75 m EACH and there are 8 of them (6 m). Hi-Link's own worked
example -- gates 3 and 4 covering "2.25-3.75m" -- fixes gate N as
0.75*N .. 0.75*(N+1), which is what ``gate_for_metres`` implements. Setting
the max gate to cover only the room stops the radar holding presence ON from
a corridor through a stud wall, which it will otherwise happily do.

WHAT IS COMPILE-TIME AND WHAT IS NOT. The pins, the throttle and the entity
names are baked into the firmware; the max gates and the absence delay live
in the LD2410's own NVM and are set over HTTP at runtime. ``tune`` therefore
needs no re-flash, and it READS BACK every value it writes rather than
claiming success -- an ESPHome rename would otherwise fail silently.

THE ENTITY NAME STAYS "Presence" IN EVERY ROOM. web_server derives the URL
path from the name, so renaming it to "Office presence" moves the endpoint to
/binary_sensor/office_presence and silently breaks presence.room_sensor_url.
The ROOM goes in the device name (jarvis-office), which is the hostname and
the OTA target, and never in the entity.

SECRETS DO NOT GO IN THE REPO. The profile (with the Wi-Fi PSK) lives in
~/.config/jarvis/room-sensors/<room>.json at 0600, and the rendered YAML in
the same directory's build/ -- never beside the template, which keeps its
placeholders. Nothing here prints the password.

THE ONE THING THIS SCRIPT CANNOT DO FOR YOU is grant itself the serial port.
/dev/ttyUSB0 is root:dialout 0660 and this account is not in dialout
(measured 2026-09-03: open() -> EACCES). ``doctor`` prints the one-line fix
and the two ways round it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "esphome" / "jarvis-room-sensor.yaml"
HOME_CFG = Path.home() / ".config" / "jarvis"
PROFILE_DIR = HOME_CFG / "room-sensors"
BUILD_DIR = PROFILE_DIR / "build"
ESPHOME_VENV = Path.home() / "esphome-venv"
ASSISTANT_JSON = HOME_CFG / "assistant.json"

# ---------------------------------------------------------------- the radar
# Every one of these is Hi-Link's, not mine. docs/room-sensor-bedroom.md §0
# carries the citations and the gate-boundary argument.
GATE_M = 0.75                 # one distance gate
MAX_GATE = 8                  # gates 0..8, so 6.0 m
MAX_RANGE_M = GATE_M * MAX_GATE
BLIND_M = 0.75                # nothing at all is detected inside this
STILL_FLOOR_M = 1.5           # no STILL-target detection inside this (gates 0,1)
# The LD2410's factory absence delay. Raise it if a still target flickers off
# while he reads; it is a DEPARTURE knob and costs nothing on arrival.
DEFAULT_TIMEOUT_S = 5

# ------------------------------------------------------------- the network
SPARK_SUBNET_HINT = "192.168.50."
DEFAULT_GATEWAY = "192.168.50.1"
DEFAULT_SUBNET = "255.255.255.0"

# ------------------------------------------------------------- ESPHome bits
# web_server derives an object_id from the entity's name: lower-cased, every
# run of non-alphanumerics collapsed to one underscore. These are the names in
# the template, and the URLs they produce.
PRESENCE_ENTITY = "Presence"
NUMBER_ENTITIES = {                     # profile field -> entity name
    "timeout_s": "Absence delay",
    "max_move_gate": "Max move gate",
    "max_still_gate": "Max still gate",
}


def object_id(name: str) -> str:
    """ESPHome's own sanitize: lower-case, non-alphanumerics -> '_'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def gate_for_metres(metres: float) -> int:
    """The smallest gate whose band REACHES ``metres``.

    Gate N spans 0.75*N .. 0.75*(N+1), so 3.0 m needs gate 4 (3.0-3.75), not
    gate 3 (2.25-3.0) which stops exactly at it. Clamped to 1..8: gate 0 alone
    is 0-0.75 m, which is inside the blind zone and therefore never useful.
    """
    if metres <= 0:
        raise ValueError("a distance must be positive")
    return max(1, min(MAX_GATE, math.ceil(float(metres) / GATE_M)))


# ------------------------------------------------------------------ presets
# A preset is a MOUNT, not a room. It answers: where does the thing physically
# sit, how far away is the nearest body it must hold, and how far is the far
# edge of what it should see. `still` False means this position is honestly a
# motion sensor and the config says so instead of pretending.
@dataclass(frozen=True)
class Preset:
    key: str
    blurb: str
    mount: str
    nearest_m: float
    range_m: float
    still: bool = True
    timeout_s: int = DEFAULT_TIMEOUT_S


PRESETS = {p.key: p for p in (
    Preset(
        key="desk",
        blurb="he sits still at a desk; the radar watches the CHAIR from across the room",
        mount=("On a shelf or wall ACROSS the room from the chair, 1.0-1.5 m up, "
               "pointed at where his chest is when he is sitting. NOT on the "
               "monitor: at 0.6 m the chair is inside the 1.5 m still floor and "
               "he vanishes the moment he stops fidgeting -- the exact failure "
               "this sensor was bought to avoid."),
        nearest_m=2.0, range_m=3.5, timeout_s=10),
    Preset(
        key="bedroom",
        blurb="holds a sleeper: ceiling, flat, above the sternum",
        mount=("Ceiling, face parallel to it, pointing straight down, directly "
               "over your sternum (~0.6-0.7 m out from the headboard wall, on "
               "the bed centreline; over YOUR side on a shared bed). Ceiling "
               "height minus mattress-plus-0.2 m must be >= 1.55 m. "
               "See docs/room-sensor-bedroom.md -- the nightstand does not work."),
        nearest_m=1.6, range_m=2.5, timeout_s=30),
    Preset(
        key="room",
        blurb="general living space: is anybody in here",
        mount=("1.0-1.5 m up, on a wall or shelf, pointed ACROSS the room like a "
               "small speaker. Keep the far gate inside the room: the LD2410 "
               "reads through plasterboard and will hold presence ON from the "
               "corridor."),
        nearest_m=1.6, range_m=4.5, timeout_s=10),
    Preset(
        key="doorway",
        blurb="a threshold: motion only, and it says so",
        mount=("Beside or above the doorway, pointed along the walk-through. "
               "Anything this close is a MOTION sensor -- there is no still "
               "detection inside 1.5 m -- which is fine for a threshold and "
               "wrong for a chair."),
        nearest_m=0.8, range_m=2.0, still=False, timeout_s=3),
)}


# ------------------------------------------------------------------ profile
@dataclass
class Profile:
    """One sensor, in one room. Everything ``flash`` and ``tune`` need."""
    room: str
    ssid: str = ""
    password: str = ""
    ota_password: str = ""
    ip: str = ""
    gateway: str = DEFAULT_GATEWAY
    subnet: str = DEFAULT_SUBNET
    board: str = "esp32dev"
    rx_pin: str = "GPIO16"          # <- LD2410 TX  (crossed)
    tx_pin: str = "GPIO17"          # -> LD2410 RX  (crossed)
    preset: str = "room"
    mount_note: str = ""
    nearest_m: float = 1.6
    range_m: float = 4.5
    still: bool = True
    timeout_s: int = DEFAULT_TIMEOUT_S
    serial_port: str = "/dev/ttyUSB0"
    flashed: bool = False           # set once a USB flash has succeeded
    notes: str = ""

    # ---- derived -------------------------------------------------------
    @property
    def device_name(self) -> str:
        """The hostname and OTA target. The ROOM lives here, not in the
        entity name -- see the module docstring."""
        return "jarvis-" + re.sub(r"[^a-z0-9]+", "-", self.room.lower()).strip("-")

    @property
    def max_move_gate(self) -> int:
        return gate_for_metres(self.range_m)

    @property
    def max_still_gate(self) -> int:
        """Still gates cannot start before 1.5 m, so a still-capable mount
        never asks for less than gate 2."""
        return max(2, self.max_move_gate) if self.still else self.max_move_gate

    @property
    def base_url(self) -> str:
        return "http://%s" % self.ip

    @property
    def presence_url(self) -> str:
        return "%s/binary_sensor/%s" % (self.base_url, object_id(PRESENCE_ENTITY))

    # ---- storage -------------------------------------------------------
    @classmethod
    def path_for(cls, room: str) -> Path:
        return PROFILE_DIR / ("%s.json" % re.sub(r"[^a-z0-9-]+", "-", room.lower()).strip("-"))

    @classmethod
    def load(cls, room: str) -> "Profile":
        path = cls.path_for(room)
        if not path.exists():
            raise SystemExit("no profile for %r yet. Run:  %s init %s --help"
                             % (room, sys.argv[0], room))
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> Path:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = self.path_for(self.room)
        blob = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        # 0600 BEFORE the write, not after: the PSK must never exist on disk
        # world-readable, not even for the microsecond between the two calls.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return path

    def describe(self) -> str:
        near, far = self.nearest_m, self.range_m
        lines = [
            "  room            %s   (device %s)" % (self.room, self.device_name),
            "  address         %s   gateway %s" % (self.ip or "(unset)", self.gateway),
            "  mount preset    %s -- %s" % (self.preset,
                                            PRESETS[self.preset].blurb
                                            if self.preset in PRESETS else "custom"),
            "  watches         %.2f m to %.2f m" % (near, far),
            "  gates           move <= %d (%.2f m), still <= %d (%.2f m)"
            % (self.max_move_gate, self.max_move_gate * GATE_M,
               self.max_still_gate, self.max_still_gate * GATE_M),
            "  still presence  %s" % ("yes" if self.still
                                      else "NO -- motion only (nearest %.2f m < %.2f m)"
                                           % (near, STILL_FLOOR_M)),
            "  absence delay   %d s" % self.timeout_s,
            "  wiring          LD2410 TX -> %s, RX -> %s (crossed), VCC -> 5V"
            % (self.rx_pin, self.tx_pin),
            "  presence url    %s" % (self.presence_url if self.ip else "(needs --ip)"),
        ]
        if self.mount_note:
            lines.append("  note            %s" % self.mount_note)
        return "\n".join(lines)


# ------------------------------------------------------------------ geometry
def check_geometry(nearest_m: float, range_m: float, still: bool) -> list:
    """Every reason this mount will disappoint him, in his words.

    Returns a list of (level, message). ``level`` "stop" means init refuses
    without --motion-only; "warn" is printed and continued.
    """
    out = []
    if nearest_m < BLIND_M:
        out.append(("stop", "%.2f m is inside the module's blind zone (%.2f m). "
                            "It will not see anything there at all."
                            % (nearest_m, BLIND_M)))
    elif nearest_m < STILL_FLOOR_M and still:
        out.append(("stop", "%.2f m is inside the 1.5 m STILL floor: gates 0 and 1 "
                            "have no rest sensitivity, so a body that stops moving "
                            "disappears. That is a PIR, which is what you were "
                            "avoiding. Move the sensor further from the chair/bed, "
                            "or pass --motion-only if this really is a doorway."
                            % nearest_m))
    if range_m > MAX_RANGE_M:
        out.append(("warn", "%.2f m is past the module's %.1f m ceiling; the far "
                            "gate will be clamped to %d (%.2f m)."
                            % (range_m, MAX_RANGE_M, MAX_GATE, MAX_RANGE_M)))
    if range_m <= nearest_m:
        out.append(("stop", "the far edge (%.2f m) must be beyond the nearest body "
                            "(%.2f m)." % (range_m, nearest_m)))
    if range_m >= 5.0:
        out.append(("warn", "at %.2f m the far gate reaches through most stud walls. "
                            "If presence sticks ON when the room is empty, lower "
                            "--range until it stops seeing the corridor."
                            % range_m))
    return out


# ------------------------------------------------------------------ preflight
def _port_state(port: str) -> tuple:
    """(exists, writable, detail) for a serial port, without opening a shell."""
    p = Path(port)
    if not p.exists():
        return False, False, "not present"
    try:
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except PermissionError as exc:
        return True, False, str(exc)
    except OSError as exc:
        return True, False, str(exc)
    os.close(fd)
    return True, True, "writable"


def esphome_bin() -> Optional[Path]:
    for cand in (ESPHOME_VENV / "bin" / "esphome",
                 Path.home() / ".local" / "bin" / "esphome"):
        if cand.exists():
            return cand
    found = shutil.which("esphome")
    return Path(found) if found else None


def ping(host: str, timeout_s: float = 1.0) -> bool:
    try:
        return subprocess.run(["ping", "-c", "1", "-W", str(int(max(1, timeout_s))), host],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout_s + 2).returncode == 0
    except Exception:
        return False


def cmd_doctor(args) -> int:
    print("ESP32 room sensor -- preflight\n")
    ok = True

    port = args.port
    exists, writable, detail = _port_state(port)
    if not exists:
        ok = False
        print("[ ] serial port %s: NOT PRESENT" % port)
        print("      Plug the ESP32 in by USB, then:  ls /dev/ttyUSB* /dev/ttyACM*")
        print("      A board with no driver shows nothing here; `lsusb` should list")
        print("      a CP210x, CH340 or FTDI bridge.")
    elif not writable:
        ok = False
        print("[ ] serial port %s: PRESENT but not writable by you (%s)" % (port, detail))
        print("      /dev/ttyUSB0 is root:dialout 0660 and you are not in dialout.")
        print("      THE FIX, once, in your own terminal (it needs your password):")
        print("          sudo usermod -aG dialout $USER")
        print("      then LOG OUT and back in (a new shell alone is not enough -- the")
        print("      group is stamped on the session), and re-run this.")
        print("      Or, without the group: flash with sudo, once --")
        print("          sudo -E %s flash %s" % (sys.argv[0], args.room or "<room>"))
        print("      Or flash from HPCOMPUTER's browser: `%s build <room>` here, then")
        print("      open https://web.esphome.io there and install the .bin it prints.")
    else:
        print("[x] serial port %s: writable" % port)

    binary = esphome_bin()
    if binary is None:
        ok = False
        print("[ ] esphome: NOT INSTALLED")
        print("      Install it OUTSIDE ~/vss_env (that venv belongs to VSS):")
        if (Path.home() / ".local" / "bin" / "uv").exists():
            print("          ~/.local/bin/uv venv %s && \\" % ESPHOME_VENV)
            print("            ~/.local/bin/uv pip install --python %s/bin/python esphome"
                  % ESPHOME_VENV)
        else:
            print("          python3 -m venv %s && %s/bin/pip install esphome"
                  % (ESPHOME_VENV, ESPHOME_VENV))
        print("      Or let this script do it:  %s install-esphome" % sys.argv[0])
    else:
        print("[x] esphome: %s" % binary)

    if not TEMPLATE.exists():
        ok = False
        print("[ ] template %s: MISSING" % TEMPLATE)
    else:
        print("[x] template: %s" % TEMPLATE)

    print("[x] this box: 192.168.50.109/24 via %s (your sensor must be on 192.168.50.x)"
          % DEFAULT_GATEWAY)

    if args.room:
        try:
            prof = Profile.load(args.room)
        except SystemExit as exc:
            print("[ ] profile %r: %s" % (args.room, exc))
            return 1
        print("\nProfile %r:\n%s" % (args.room, prof.describe()))
        if prof.ip:
            print("\n[%s] %s answers ping" % ("x" if ping(prof.ip) else " ", prof.ip))
    print("\n%s" % ("Ready to flash." if ok else "Fix the [ ] lines above first."))
    return 0 if ok else 1


def cmd_install_esphome(args) -> int:
    """ESPHome in its own venv. Never ~/vss_env -- that one is shared with VSS
    and a PlatformIO toolchain does not belong in it."""
    if esphome_bin() and not args.force:
        print("esphome already at %s (use --force to reinstall)" % esphome_bin())
        return 0
    uv = Path.home() / ".local" / "bin" / "uv"
    if uv.exists():
        subprocess.run([str(uv), "venv", str(ESPHOME_VENV)], check=True)
        subprocess.run([str(uv), "pip", "install", "--python",
                        str(ESPHOME_VENV / "bin" / "python"), "esphome"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(ESPHOME_VENV)], check=True)
        subprocess.run([str(ESPHOME_VENV / "bin" / "pip"), "install", "esphome"], check=True)
    print("esphome installed: %s" % (ESPHOME_VENV / "bin" / "esphome"))
    return 0


# ---------------------------------------------------------------------- init
def cmd_init(args) -> int:
    preset = PRESETS.get(args.preset)
    if preset is None:
        raise SystemExit("unknown preset %r; choose from: %s"
                         % (args.preset, ", ".join(sorted(PRESETS))))
    try:
        prof = Profile.load(args.room)
        print("updating the existing profile for %r" % args.room)
    except SystemExit:
        prof = Profile(room=args.room)

    prof.preset = preset.key
    prof.mount_note = preset.mount
    prof.nearest_m = args.nearest if args.nearest is not None else preset.nearest_m
    prof.range_m = args.range if args.range is not None else preset.range_m
    prof.timeout_s = args.timeout if args.timeout is not None else preset.timeout_s
    prof.still = preset.still and not args.motion_only
    for attr, val in (("ssid", args.ssid), ("password", args.password),
                      ("ota_password", args.ota_password), ("ip", args.ip),
                      ("gateway", args.gateway), ("subnet", args.subnet),
                      ("board", args.board), ("serial_port", args.port),
                      ("notes", args.note)):
        if val:
            setattr(prof, attr, val)
    if args.wrover:
        prof.board, prof.rx_pin, prof.tx_pin = "esp-wrover-kit", "GPIO32", "GPIO33"

    problems = check_geometry(prof.nearest_m, prof.range_m, prof.still)
    for level, msg in problems:
        print(("STOP: " if level == "stop" else "note: ") + msg)
    if any(lvl == "stop" for lvl, _ in problems) and not args.motion_only:
        return 2
    if not prof.still:
        print("note: still-presence is OFF for this mount; Jarvis will see motion "
              "only, and the config records that rather than implying otherwise.")

    if not prof.ota_password:
        # Not a secret that protects anything of his -- it stops a neighbour
        # reflashing the device -- but a blank one disables OTA entirely.
        prof.ota_password = "jarvis-" + prof.device_name
    if prof.ip and not prof.ip.startswith(SPARK_SUBNET_HINT):
        print("note: %s is not on this box's subnet (%sx). Jarvis reaches the "
              "sensor over plain HTTP on the LAN, so they must share one."
              % (prof.ip, SPARK_SUBNET_HINT))
    if prof.ip and ping(prof.ip):
        print("note: %s already answers ping. If that is not this sensor, pick "
              "another address -- a collision looks exactly like a dead sensor."
              % prof.ip)

    missing = [k for k in ("ssid", "password", "ip") if not getattr(prof, k)]
    path = prof.save()
    print("\nprofile written: %s (0600)\n%s" % (path, prof.describe()))
    if missing:
        print("\nstill needed before flashing: %s" % ", ".join("--" + m for m in missing))
        return 1
    print("\nMount it as follows before you trust the numbers:\n  %s"
          % preset.mount.replace(". ", ".\n  "))
    print("\nNext:  %s flash %s" % (sys.argv[0], prof.room))
    return 0


# -------------------------------------------------------------------- render
def render_yaml(prof: Profile) -> str:
    """The template with this room's substitutions, and nothing else changed.

    Rewriting only the substitutions block keeps every decision the template
    documents -- no api: block, has_target for presence, the 250 ms throttle
    -- exactly where its comments say it is.
    """
    text = TEMPLATE.read_text()
    subs = {
        "device_name": prof.device_name,
        "wifi_ssid": prof.ssid,
        "wifi_password": prof.password,
        "ota_password": prof.ota_password,
        "static_ip": prof.ip,
        "gateway": prof.gateway,
        "subnet": prof.subnet,
    }
    out = []
    in_subs = False
    for line in text.splitlines():
        if line.startswith("substitutions:"):
            in_subs, _ = True, out.append(line)
            for key, val in subs.items():
                out.append("  %s: %s" % (key, json.dumps(val)))
            out.append("  # rendered by scripts/room_sensor.py for room %r" % prof.room)
            continue
        if in_subs:
            if line and not line[0].isspace():
                in_subs = False          # the block ended; fall through
            else:
                continue                 # drop the template's placeholder lines
        out.append(line)
    body = "\n".join(out) + "\n"
    if prof.board != "esp32dev":
        body = body.replace("  board: esp32dev", "  board: %s" % prof.board, 1)
    if prof.rx_pin != "GPIO16":
        body = body.replace("  rx_pin: GPIO16", "  rx_pin: %s" % prof.rx_pin, 1)
    if prof.tx_pin != "GPIO17":
        body = body.replace("  tx_pin: GPIO17", "  tx_pin: %s" % prof.tx_pin, 1)
    return body


def build_path(prof: Profile) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return BUILD_DIR / ("%s.yaml" % prof.device_name)


def write_build(prof: Profile) -> Path:
    path = build_path(prof)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(render_yaml(prof))
    return path


def cmd_build(args) -> int:
    prof = Profile.load(args.room)
    for key in ("ssid", "password", "ip"):
        if not getattr(prof, key):
            raise SystemExit("profile %r has no %s; re-run init with --%s"
                             % (prof.room, key, key))
    path = write_build(prof)
    print("rendered: %s (0600, contains your Wi-Fi password -- not in the repo)" % path)
    binary = esphome_bin()
    if binary is None:
        print("esphome is not installed; run:  %s install-esphome" % sys.argv[0])
        return 1
    print("validating...")
    res = subprocess.run([str(binary), "config", str(path)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print("\nesphome rejected the config. If it named one of the `number:` "
              "entries, that component was renamed upstream -- delete that block "
              "from the template; nothing Jarvis reads depends on it.")
        return 1
    print("config valid.")
    if args.compile:
        res = subprocess.run([str(binary), "compile", str(path)])
        if res.returncode != 0:
            return res.returncode
        for name in ("firmware.factory.bin", "firmware.bin"):
            hits = sorted((path.parent / ".esphome").rglob(name))
            if hits:
                print("\nfirmware: %s" % hits[-1])
                print("To flash from another machine's browser: serve this directory")
                print("  (cd %s && python3 -m http.server 8000)" % hits[-1].parent)
                print("  then open https://web.esphome.io on that machine and install")
                print("  %s from http://192.168.50.109:8000/" % name)
                break
    return 0


# --------------------------------------------------------------------- flash
def cmd_flash(args) -> int:
    prof = Profile.load(args.room)
    binary = esphome_bin()
    if binary is None:
        raise SystemExit("esphome is not installed; run:  %s install-esphome" % sys.argv[0])
    path = write_build(prof)

    over_air = args.ota or (prof.flashed and not args.usb)
    if over_air:
        if not ping(prof.ip):
            print("note: %s does not answer ping; an OTA will fail. Plug it in and "
                  "use --usb if it never joined the Wi-Fi." % prof.ip)
        target = prof.ip
    else:
        exists, writable, detail = _port_state(prof.serial_port)
        if not exists:
            raise SystemExit("%s is not there. Plug the ESP32 in, or pass --ota."
                             % prof.serial_port)
        if not writable:
            print("cannot open %s: %s" % (prof.serial_port, detail))
            print("Run `%s doctor` for the one-line fix (you are not in dialout)."
                  % sys.argv[0])
            return 13
        target = prof.serial_port

    print("flashing %s via %s ..." % (prof.device_name, target))
    res = subprocess.run([str(binary), "run", str(path), "--device", target,
                          "--no-logs" if args.no_logs else "--verbose"][:4 + (0 if args.no_logs else 0)])
    if res.returncode != 0:
        print("\nesphome exited %d." % res.returncode)
        if not over_air:
            print("If it timed out waiting for the bootloader: hold the board's BOOT "
                  "button while it says 'Connecting...', release when it starts writing.")
        return res.returncode

    if not over_air:
        prof.flashed = True
        prof.save()
    print("\nflashed. Now:")
    print("  %s tune %s     # push the mount geometry into the radar" % (sys.argv[0], prof.room))
    print("  %s check %s    # what Jarvis would read from it" % (sys.argv[0], prof.room))
    return 0


# ---------------------------------------------------------------------- tune
def _http_get(url: str, timeout: float = 3.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:   # noqa: S310 - LAN, plain http by design
        return resp.read().decode("utf-8", "replace")


def _number_get(prof: Profile, entity: str, timeout: float = 3.0):
    body = _http_get("%s/number/%s" % (prof.base_url, object_id(entity)), timeout)
    try:
        return json.loads(body).get("value")
    except Exception:
        return None


def _number_set(prof: Profile, entity: str, value, timeout: float = 3.0) -> None:
    url = "%s/number/%s/set?value=%s" % (prof.base_url, object_id(entity), value)
    req = urllib.request.Request(url, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=timeout):            # noqa: S310
        pass


def cmd_tune(args) -> int:
    """Write the geometry into the radar's own NVM, then READ IT BACK.

    Read-back is the whole point: these entities are named in the template,
    and an ESPHome rename would otherwise leave the write silently ignored
    and this script claiming a tuning it never applied.
    """
    prof = Profile.load(args.room)
    if not prof.ip:
        raise SystemExit("profile %r has no --ip yet" % prof.room)
    wanted = {
        "Absence delay": prof.timeout_s,
        "Max move gate": prof.max_move_gate,
        "Max still gate": prof.max_still_gate,
    }
    print("%s -> %s" % (prof.device_name, prof.base_url))
    bad = 0
    for entity, value in wanted.items():
        try:
            before = _number_get(prof, entity)
            _number_set(prof, entity, value)
            time.sleep(0.4)              # the module writes its own NVM
            after = _number_get(prof, entity)
        except urllib.error.URLError as exc:
            print("  %-14s UNREACHABLE (%s)" % (entity, exc.reason))
            bad += 1
            continue
        except Exception as exc:                            # noqa: BLE001
            print("  %-14s failed: %s" % (entity, exc))
            bad += 1
            continue
        agree = after is not None and abs(float(after) - float(value)) < 0.51
        print("  %-14s %s -> %s  %s" % (entity, before, after,
                                        "ok" if agree else "NOT APPLIED (asked %s)" % value))
        bad += 0 if agree else 1
    if bad:
        print("\n%d setting(s) did not take. Open %s/ in a browser and set them by "
              "hand; the names there are the truth." % (bad, prof.base_url))
        return 1
    print("\ntuned. %.2f m of coverage, still-presence %s."
          % (prof.max_still_gate * GATE_M, "on" if prof.still else "off (motion only)"))
    return 0


# --------------------------------------------------------------- check/watch
def _jarvis_reader(prof: Profile):
    """Read through jarvis.roomsensor, not a hand-rolled parser: the point of
    the check is what JARVIS will see, including its URL normalising and its
    tolerance of the body ESPHome actually sends."""
    sys.path.insert(0, str(REPO))
    from jarvis import roomsensor                      # noqa: PLC0415 - after sys.path
    url = roomsensor.normalize_url(prof.presence_url)

    def read():
        return roomsensor.parse_state(_http_get(url, timeout=2.0))
    return url, read


def cmd_check(args) -> int:
    prof = Profile.load(args.room)
    if not prof.ip:
        raise SystemExit("profile %r has no --ip yet" % prof.room)
    url, read = _jarvis_reader(prof)
    print("url Jarvis would use: %s" % url)
    try:
        t0 = time.monotonic()
        state = read()
        ms = (time.monotonic() - t0) * 1000
    except Exception as exc:                                     # noqa: BLE001
        print("UNREACHABLE: %s" % exc)
        print("  * is the device on? (%s)" % ("ping ok" if ping(prof.ip) else "ping FAILS"))
        print("  * is %s the address you flashed?" % prof.ip)
        return 1
    if state is None:
        print("reachable, but the body did not parse as a presence state.")
        return 1
    print("presence: %s   (%.0f ms)" % ("SOMEONE" if state else "empty", ms))
    print("\nIf that says SOMEONE with the room empty, the far gate is seeing "
          "through a wall: lower --range and re-run tune.")
    return 0


def cmd_watch(args) -> int:
    prof = Profile.load(args.room)
    url, read = _jarvis_reader(prof)
    print("watching %s -- walk out, wait, walk back in. Ctrl-C to stop.\n" % url)
    last, since = None, time.monotonic()
    try:
        while True:
            try:
                state = read()
            except Exception:                                    # noqa: BLE001
                state = None
            if state != last:
                now = time.monotonic()
                label = {True: "SOMEONE", False: "empty", None: "unreachable"}[state]
                print("%s  %-11s  after %5.1fs" % (time.strftime("%H:%M:%S"), label,
                                                   now - since))
                last, since = state, now
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


# ------------------------------------------------------------ install-config
def cmd_install_config(args) -> int:
    """Write presence.* into assistant.json, with a backup and no guessing.

    A hand edit to assistant.json does NOTHING until Jarvis restarts --
    reload_if_changed() has no callers -- so this says so rather than leaving
    him waiting for a sensor that is configured and unread.
    """
    prof = Profile.load(args.room)
    if not prof.ip:
        raise SystemExit("profile %r has no --ip yet" % prof.room)
    if not ASSISTANT_JSON.exists():
        raise SystemExit("%s does not exist" % ASSISTANT_JSON)
    data = json.loads(ASSISTANT_JSON.read_text())
    presence = dict(data.get("presence") or {})
    before = dict(presence)
    presence["room_sensor_enabled"] = True
    presence["room_sensor_url"] = prof.presence_url
    presence.setdefault("room_sensor_timeout_s", 1.5)
    if args.fast:
        presence["poll_s_away"] = 5
    if before == presence:
        print("no change needed: assistant.json already says exactly this.")
        return 0
    print("presence block:")
    for key in sorted(set(before) | set(presence)):
        if before.get(key) != presence.get(key):
            print("  %-24s %r -> %r" % (key, before.get(key), presence.get(key)))
    if not args.yes:
        print("\nnothing written (pass --yes to apply).")
        return 0
    backup = ASSISTANT_JSON.with_suffix(".json.bak-roomsensor-%s"
                                        % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(ASSISTANT_JSON, backup)
    data["presence"] = presence
    ASSISTANT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print("\nwritten (backup: %s)" % backup.name)
    print("RESTART JARVIS -- a config edit does nothing until it starts again.")
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="room_sensor.py", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets:\n" + "\n".join(
            "  %-9s %s" % (p.key, p.blurb) for p in PRESETS.values()))
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="what is missing, and the exact fix")
    d.add_argument("room", nargs="?")
    d.add_argument("--port", default="/dev/ttyUSB0")
    d.set_defaults(func=cmd_doctor)

    i = sub.add_parser("install-esphome", help="esphome in its own venv")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_install_esphome)

    n = sub.add_parser("init", help="create or update a room's profile")
    n.add_argument("room")
    n.add_argument("--preset", default="room", choices=sorted(PRESETS))
    n.add_argument("--ssid"), n.add_argument("--password")
    n.add_argument("--ota-password")
    n.add_argument("--ip"), n.add_argument("--gateway"), n.add_argument("--subnet")
    n.add_argument("--nearest", type=float,
                   help="metres to the NEAREST body it must hold (the chair, the bed)")
    n.add_argument("--range", type=float,
                   help="metres to the far edge of what it should watch")
    n.add_argument("--timeout", type=int, help="absence delay in seconds")
    n.add_argument("--motion-only", action="store_true",
                   help="accept a mount inside the 1.5 m still floor (a doorway)")
    n.add_argument("--board", help="platformio board id (default esp32dev)")
    n.add_argument("--wrover", action="store_true",
                   help="ESP32-WROVER: PSRAM owns GPIO16/17, use 32/33")
    n.add_argument("--port", default="/dev/ttyUSB0")
    n.add_argument("--note", default="")
    n.set_defaults(func=cmd_init)

    b = sub.add_parser("build", help="render + validate (and optionally compile)")
    b.add_argument("room")
    b.add_argument("--compile", action="store_true",
                   help="also compile, and print the .bin for a browser flash")
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("flash", help="USB the first time, OTA after")
    f.add_argument("room")
    f.add_argument("--usb", action="store_true", help="force the serial port")
    f.add_argument("--ota", action="store_true", help="force over-the-air")
    f.add_argument("--no-logs", action="store_true")
    f.set_defaults(func=cmd_flash)

    t = sub.add_parser("tune", help="push the geometry into the radar, and read it back")
    t.add_argument("room")
    t.set_defaults(func=cmd_tune)

    c = sub.add_parser("check", help="what Jarvis would read, right now")
    c.add_argument("room")
    c.set_defaults(func=cmd_check)

    w = sub.add_parser("watch", help="every transition, timed")
    w.add_argument("room")
    w.add_argument("--interval", type=float, default=0.25)
    w.set_defaults(func=cmd_watch)

    g = sub.add_parser("install-config", help="write presence.* into assistant.json")
    g.add_argument("room")
    g.add_argument("--yes", action="store_true", help="actually write it")
    g.add_argument("--fast", action="store_true",
                   help="also set poll_s_away 5 (arrival 2.7 s mean instead of 5.7)")
    g.set_defaults(func=cmd_install_config)

    s = sub.add_parser("show", help="print a profile")
    s.add_argument("room")
    s.set_defaults(func=lambda a: (print(Profile.load(a.room).describe()), 0)[1])
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
