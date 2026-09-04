#!/usr/bin/env python3
"""Which mode the camera actually GRANTS, and how fast it hands frames over
-- measured in numbers, without decoding a single one of them.

    ~/vss_env/bin/python scripts/camera_mode_probe.py
    ~/vss_env/bin/python scripts/camera_mode_probe.py --grabs 60 --device 0

WHY THIS EXISTS. The preview pane's live log said the device delivered
~7.5 fps at 1280x720 whatever ``camera.preview_fps`` asked for (6.0 fps at 6
requested, 7.5 at 10, 7.4-7.6 at 15), so raising the request did nothing for
the lag he can see. Two explanations were never measured: the driver may not
have GRANTED the MJPG format the app asks for (a 720p YUYV stream on USB 2.0
is bandwidth-capped near 7.5 fps, and v4l2 does not refuse a format request
-- it quietly grants something else), or the 720p mode is capped whatever
the format. Nothing in the running app reads the granted mode back. This
script does, for each of the four modes that matter, and it times the
device's OWN delivery rate.

WHAT IT DOES NOT DO, MECHANICALLY. Hunter: *"i dont want you to look at
anything the camera sees without my explicit permission."* This script
opens the device, asks for a mode, reads back the properties the driver
granted, and times ``VideoCapture.grab()`` -- which DEQUEUES a buffer
without decoding it. There is no ``read``, no ``retrieve``, no ``imshow``,
no ``imwrite``, and no variable in this file ever holds pixel data. The
first thing ``main`` does is tokenize this file and refuse to run if its
CODE (strings and comments excluded) names any of those calls;
tests/test_camera_mode_probe.py pins the same rule from outside. Every
line printed is a name, a size, a rate or a millisecond.

It runs each configuration TWICE, setting the FOURCC before the size and
then after it, because OpenCV's V4L2 backend has at various versions
re-negotiated the format when the size changed -- if one order is honoured
and the other is not, that is the fix and this is where it shows.

If the device is busy, absent, or will not grab, it says so once and
stops. It does not retry, and it does not try other /dev/video nodes: the
app opens index 0 and that is the only question here.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import tokenize
from pathlib import Path

# The calls that would turn a frame into something a person could look at.
# Checked as NAME tokens in this file's own code, so naming them in a
# docstring (as this paragraph does) is fine and calling one is fatal.
PIXEL_CALLS = frozenset({
    "read", "retrieve", "imshow", "imwrite", "imencode", "imdecode",
    "tofile", "save", "frombuffer", "namedWindow", "waitKey",
})

# What the app asks for, and the three modes that settle whether the answer
# is the format or the bandwidth. (width, height, fourcc); 30 fps requested
# for each.
MODES = ((1280, 720, "MJPG"), (1280, 720, "YUYV"),
         (640, 480, "MJPG"), (640, 480, "YUYV"))
ORDERS = ("fourcc-first", "size-first")
REQUEST_FPS = 30.0


def self_check(path: str = __file__) -> list:
    """The names in this file's CODE that are on the forbidden list. Empty
    means the file may run. Strings and comments are skipped, which is why
    the docstring above can discuss ``read`` without tripping it."""
    hits = []
    source = Path(path).read_bytes()
    for tok in tokenize.tokenize(io.BytesIO(source).readline):
        if tok.type == tokenize.NAME and tok.string in PIXEL_CALLS:
            hits.append("%s line %d" % (tok.string, tok.start[0]))
    return hits


def fourcc_name(value) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return ""
    if code <= 0:
        return ""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def granted(cv2, cap) -> dict:
    """The properties the driver actually granted, as numbers."""
    out = {}
    for name, prop in (("width", cv2.CAP_PROP_FRAME_WIDTH),
                       ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                       ("fps", cv2.CAP_PROP_FPS),
                       ("buffersize", cv2.CAP_PROP_BUFFERSIZE),
                       ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
                       ("exposure", cv2.CAP_PROP_EXPOSURE)):
        try:
            out[name] = float(cap.get(prop))
        except Exception:  # noqa: BLE001 - an unsupported property is data
            out[name] = -1.0
    try:
        out["fourcc"] = fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))
    except Exception:  # noqa: BLE001
        out["fourcc"] = ""
    return out


def request(cv2, cap, width: int, height: int, fourcc: str,
            order: str) -> None:
    """Ask for a mode in one of the two orders. The return values of
    ``set`` are not trusted -- the read-back afterwards is the truth."""
    def set_fourcc():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    def set_size():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    if order == "fourcc-first":
        set_fourcc()
        set_size()
    else:
        set_size()
        set_fourcc()
    cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)


def time_grabs(cap, warmup: int, grabs: int) -> dict:
    """Dequeue ``warmup + grabs`` buffers and time the last ``grabs``.

    ``grab()`` is the only call made on the device here. It returns a bool;
    nothing it dequeued is retrieved, so no pixel ever reaches this
    process's Python side.
    """
    failed = 0
    for _ in range(warmup):
        if not cap.grab():
            failed += 1
            if failed >= 5:
                return {"ok": False, "failed": failed, "fps": 0.0,
                        "p50_ms": 0.0, "max_ms": 0.0, "grabs": 0}
    gaps = []
    t_start = time.perf_counter()
    last = t_start
    for _ in range(grabs):
        if not cap.grab():
            failed += 1
            if failed >= 5:
                break
            continue
        now = time.perf_counter()
        gaps.append((now - last) * 1000.0)
        last = now
    elapsed = time.perf_counter() - t_start
    n = len(gaps)
    gaps.sort()
    return {"ok": n > 0, "failed": failed, "grabs": n,
            "fps": (n / elapsed) if elapsed > 0 else 0.0,
            "p50_ms": gaps[n // 2] if n else 0.0,
            "max_ms": gaps[-1] if n else 0.0}


def probe_one(cv2, target, width: int, height: int, fourcc: str,
              order: str, warmup: int, grabs: int,
              buffersize: bool) -> dict:
    """Open, ask, read back, time, release. One configuration."""
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        cap.release()
        return {"opened": False}
    try:
        backend = ""
        try:
            backend = str(cap.getBackendName())
        except Exception:  # noqa: BLE001
            pass
        request(cv2, cap, width, height, fourcc, order)
        got = granted(cv2, cap)
        buf_accepted = None
        if buffersize:
            try:
                buf_accepted = bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            except Exception:  # noqa: BLE001
                buf_accepted = False
            got_after = granted(cv2, cap)
            got["buffersize_after_set"] = got_after["buffersize"]
        timing = time_grabs(cap, warmup, grabs)
        # Read again AFTER streaming started: some drivers only settle the
        # format on the first dequeue.
        after = granted(cv2, cap)
    finally:
        cap.release()
    return {"opened": True, "backend": backend, "granted": got,
            "after": after, "buffersize_set_accepted": buf_accepted,
            "timing": timing}


def line(width, height, fourcc, order, result) -> str:
    g = result["granted"]
    a = result["after"]
    t = result["timing"]
    mode = "%dx%d %s" % (width, height, fourcc)
    got = "%.0fx%.0f %-4s %5.1f fps" % (g["width"], g["height"],
                                        g["fourcc"] or "?", g["fps"])
    settled = "" if (a["fourcc"], a["width"], a["height"]) == \
        (g["fourcc"], g["width"], g["height"]) else \
        "  (after streaming: %.0fx%.0f %s)" % (a["width"], a["height"],
                                               a["fourcc"] or "?")
    if not t["ok"]:
        rate = "grab FAILED (%d failures)" % t["failed"]
    else:
        rate = "grab %5.1f fps  p50 %6.1f ms  max %6.1f ms  (%d grabs)" % (
            t["fps"], t["p50_ms"], t["max_ms"], t["grabs"])
    return "  %-16s %-12s -> granted %s   %s%s" % (
        mode, order, got, rate, settled)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="0",
                    help="index or /dev node; the app opens 0 (default 0)")
    ap.add_argument("--grabs", type=int, default=60,
                    help="timed grabs per configuration (default 60)")
    ap.add_argument("--warmup", type=int, default=10,
                    help="untimed grabs before the clock starts (default 10)")
    ap.add_argument("--no-buffersize", action="store_true",
                    help="skip the CAP_PROP_BUFFERSIZE=1 set/read-back")
    args = ap.parse_args(argv)

    hits = self_check()
    if hits:
        print("REFUSING TO RUN: this file's code names a pixel-bearing "
              "call: %s" % ", ".join(hits))
        return 4

    try:
        # Deliberately lazy, so the self-check above runs first.
        import cv2  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print("STOPPED: cv2 is not importable (%s)" % exc)
        return 3
    target = args.device
    if isinstance(target, str) and target.isdigit():
        target = int(target)
    elif isinstance(target, str) and not Path(target).exists():
        print("STOPPED: %s does not exist" % target)
        return 3

    print("camera mode probe -- numbers only. Buffers are dequeued with "
          "grab() and never retrieved, decoded, shown or saved.")
    print("cv2 %s  device %r  %d fps requested  %d warm-up + %d timed grabs "
          "per configuration" % (cv2.__version__, target, int(REQUEST_FPS),
                                 args.warmup, args.grabs))
    print("")
    first = True
    rows = []
    for width, height, fourcc in MODES:
        for order in ORDERS:
            result = probe_one(cv2, target, width, height, fourcc, order,
                               args.warmup, args.grabs,
                               buffersize=not args.no_buffersize)
            if not result["opened"]:
                print("STOPPED: could not open device %r -- busy or absent. "
                      "Not retrying." % (target,))
                return 3
            if first:
                print("backend %s" % (result["backend"] or "?"))
                g = result["granted"]
                print("auto_exposure %.1f  exposure %.1f  (driver controls, "
                      "read only; a UVC camera on auto-exposure may lower "
                      "its frame rate in a dim room)" % (g["auto_exposure"],
                                                          g["exposure"]))
                if not args.no_buffersize:
                    print("CAP_PROP_BUFFERSIZE: was %.0f, set(1) %s, reads "
                          "back %.0f" % (
                              g["buffersize"],
                              "accepted" if result["buffersize_set_accepted"]
                              else "refused",
                              g.get("buffersize_after_set", -1.0)))
                print("")
                print("  requested        set order       granted mode"
                      "                 sustained")
                first = False
            text = line(width, height, fourcc, order, result)
            print(text)
            rows.append(text)
            if not result["timing"]["ok"]:
                print("STOPPED: the device would not grab in that mode. "
                      "Not retrying.")
                return 1
    print("")
    print("Read the 'granted' column against the 'requested' one: a fourcc "
          "that differs is a format the driver declined silently.")
    print("A sustained rate under the granted fps is the device's own cap "
          "-- the light, the exposure control or the bus, not the format.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
