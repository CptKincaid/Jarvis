#!/usr/bin/env python
"""Room sensor bench: a fake ESP32 to develop against, and two ways to
look at a real one. Stdlib only.

    # 1. no hardware yet: serve a fake ESPHome endpoint you can flip
    ~/vss_env/bin/python scripts/roomsensor_stub.py serve
    #    -> http://127.0.0.1:8781/binary_sensor/presence
    #    curl http://127.0.0.1:8781/on      # "someone walked in"
    #    curl http://127.0.0.1:8781/off     # "the room is empty"

    # 2. does Jarvis like what that URL serves? (real device or fake)
    ~/vss_env/bin/python scripts/roomsensor_stub.py check http://127.0.0.1:8781

    # 3. the hardware is on the wall: what does it see, and how fast?
    ~/vss_env/bin/python scripts/roomsensor_stub.py watch http://192.168.50.60

``check`` and ``watch`` go through the REAL jarvis/roomsensor.py -- same
parser, same timeout, same breaker -- so "check says OK" means the app
will read it too. ``serve --mode`` reproduces the three failures worth
rehearsing: garbage in the body, a host that hangs past the timeout, and a
500. Under all three Jarvis must simply fall back to the phone probe.
"""
from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.roomsensor import RoomSensor, normalize_url  # noqa: E402

HELP = """jarvis room sensor stub
  /binary_sensor/presence   the entity Jarvis reads
  /on /off /toggle          flip it
"""


class Bench:
    """The fake device's state, shared with the request handler."""

    def __init__(self, state: bool, mode: str, flip: float):
        self.state, self.mode, self.flip = state, mode, flip
        self.lock = threading.Lock()
        self.reads = 0


def _handler(bench: Bench):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: str, code: int = 200, ctype="application/json"):
            raw = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
            path = self.path.split("?")[0].rstrip("/") or "/"
            with bench.lock:
                if path in ("/on", "/off", "/toggle"):
                    bench.state = {"on": True, "off": False}.get(
                        path[1:], not bench.state)
                    return self._send(json.dumps({"state": bench.state}))
                if path != "/binary_sensor/presence":
                    return self._send(HELP + f"\nstate: {bench.state}\n",
                                      200 if path == "/" else 404, "text/plain")
                bench.reads += 1
                state, mode = bench.state, bench.mode
            if mode == "garbage":
                return self._send("<html><body>ESPHome</body></html>", 200, "text/html")
            if mode == "error":
                return self._send("", 500, "text/plain")
            if mode == "slow":
                time.sleep(5.0)          # past any sane timeout
            self._send(json.dumps({"id": "binary_sensor-presence",
                                   "value": bool(state),
                                   "state": "ON" if state else "OFF"}))

        def log_message(self, fmt, *args):
            print(f"  {time.strftime('%H:%M:%S')}  {fmt % args}", flush=True)

    return Handler


def serve(args) -> int:
    bench = Bench(args.state == "on", args.mode, args.flip)
    srv = http.server.ThreadingHTTPServer((args.host, args.port), _handler(bench))
    url = f"http://{args.host}:{args.port}/binary_sensor/presence"
    print(f"serving {url}   (mode={args.mode}, state={'ON' if bench.state else 'OFF'})")
    print("put this in ~/.config/jarvis/assistant.json:")
    print(f'    "room_sensor_enabled": true,\n    "room_sensor_url": "{url}"')
    print("flip it with:  curl -s http://%s:%d/on   (or /off, /toggle)\n"
          % (args.host, args.port))
    if bench.flip:
        def flipper():
            while True:
                time.sleep(bench.flip)
                with bench.lock:
                    bench.state = not bench.state
                print(f"  --- auto-flip -> {'ON' if bench.state else 'OFF'}", flush=True)
        threading.Thread(target=flipper, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def _read(url: str, timeout: float):
    sensor = RoomSensor(url, timeout_s=timeout)
    if not sensor.configured:
        print(f"NOT A URL: {url!r}")
        print("  needs the scheme, e.g. http://192.168.50.60")
        return None, sensor
    t0 = time.monotonic()
    value = sensor.read()
    return (value, (time.monotonic() - t0) * 1000.0), sensor


def check(args) -> int:
    got, sensor = _read(args.url, args.timeout)
    if got is None:
        return 2
    value, ms = got
    print(f"url:    {sensor.url}")
    print(f"read:   {value!r}   ({ms:.0f} ms)")
    if value is True:
        print("OK - the room says SOMEONE IS THERE.")
        print("     Jarvis would call him home on the next poll, phone or no phone.")
    elif value is False:
        print("OK - the room says NOBODY IS THERE.")
        print("     Jarvis falls through to the phone probe; an empty room on its")
        print("     own never makes him away while the phone still answers.")
    else:
        print("NO OPINION - unreachable, or the body was not something readable.")
        print("     Jarvis would behave exactly as it does with no sensor at all.")
        print("     Try:  curl -sv " + sensor.url)
        return 1
    return 0


def watch(args) -> int:
    """Print every transition with the wall clock. Walk out, wait, walk in:
    the gap printed on the ON line is your real arrival latency, sensor
    side. Add Jarvis's poll interval (presence.poll_s_away) for the rest."""
    url = normalize_url(args.url)
    if not url:
        print(f"NOT A URL: {args.url!r}")
        return 2
    sensor = RoomSensor(url, timeout_s=args.timeout)
    print(f"watching {url} every {args.interval:.1f}s -- ctrl-C to stop\n")
    last, since = "start", time.monotonic()
    try:
        while True:
            value = sensor.read()
            name = {True: "SOMEONE", False: "empty", None: "no answer"}[value]
            if name != last:
                now = time.monotonic()
                print(f"{time.strftime('%H:%M:%S')}  {name:<10} "
                      f"(after {now - since:5.1f}s of {last})", flush=True)
                last, since = name, now
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="a fake ESPHome endpoint on localhost")
    s.add_argument("--host", default="127.0.0.1")
    # 8781, not 8765: the live app's phone web UI (jarvis/webapp.py) owns
    # 8765 and is already listening on it.
    s.add_argument("--port", type=int, default=8781)
    s.add_argument("--state", choices=("on", "off"), default="off")
    s.add_argument("--mode", choices=("ok", "garbage", "slow", "error"), default="ok",
                   help="ok | garbage body | hangs past the timeout | HTTP 500")
    s.add_argument("--flip", type=float, default=0.0,
                   help="flip the state every N seconds")
    s.set_defaults(fn=serve)

    c = sub.add_parser("check", help="what would Jarvis read from this URL?")
    c.add_argument("url")
    c.add_argument("--timeout", type=float, default=1.5)
    c.set_defaults(fn=check)

    w = sub.add_parser("watch", help="print transitions with timestamps")
    w.add_argument("url")
    w.add_argument("--interval", type=float, default=0.5)
    w.add_argument("--timeout", type=float, default=1.5)
    w.set_defaults(fn=watch)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
