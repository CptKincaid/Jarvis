#!/usr/bin/env python3
"""What the queue drain costs and what it buys, in numbers, without
decoding a single frame.

    ~/vss_env/bin/python scripts/camera_drain_probe.py            # the camera
    ~/vss_env/bin/python scripts/camera_drain_probe.py --model 15 # no camera

WHY THIS EXISTS. ``jarvis/camera.DrainingCapture`` claims two things at once:
that a consumer running BELOW the device's delivered rate still gets the
newest frame, and that a consumer running AT it pays nothing for the
privilege. Both halves are claims about the device's own queue, so both have
to be measured at the device. This runs the REAL policy -- it imports
``DrainingCapture`` and calls its ``drain()``, which is the same bounded grab
loop the app runs -- at three consumer rates, and prints the delivered rate
and the drop count for each, against an undrained arm for comparison.

WHAT IT DOES NOT DO, MECHANICALLY. Hunter: *"i dont want you to look at
anything the camera sees without my explicit permission."* This script opens
the device, asks for a mode, reads back what the driver granted and times
``drain()``, which dequeues buffers and NEVER decodes one. There is no
``read``, no ``retrieve``, no ``imshow``, no ``imwrite``; no variable in this
file ever holds pixel data. The first thing ``main`` does is tokenize this
file and refuse to run if its CODE (strings and comments excluded) names any
of those calls, exactly as scripts/camera_mode_probe.py does, and
tests/test_camera_drain_probe.py pins the same rule from outside.

WHY THE RETRIEVE'S ABSENCE COSTS THE MEASUREMENT NOTHING. A drained cycle and
an undrained one decode exactly ONE buffer each -- that is the whole design.
Every difference between them is in the grab loop, and the grab loop is what
is timed here. What this cannot show is the decode cost itself, which is
identical in both arms and therefore not part of the comparison.

IT WILL NOT TAKE THE CAMERA OFF THE RUNNING JARVIS. Before opening anything
it looks for another process holding the node and stops with its pid if it
finds one. A measurement is not worth blinding the assistant. That happens
often -- Jarvis holds /dev/video0 whenever it is up -- so ``--model FPS``
runs every arm against a modelled V4L2 queue instead: the drain's own
arithmetic, real time and real sleeps, no device. It says so in its first
line, because a modelled number pasted as a measured one is exactly the
mistake this project has been damaged by.

Exit codes: 0 measured, 1 the device would not grab, 3 no camera / busy /
no cv2, 4 the self-check refused.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))   # so `import jarvis.*` works

# The calls that would turn a frame into something a person could look at.
# Checked as NAME tokens in this file's own code, so naming them in a
# docstring (as the paragraphs above do) is fine and calling one is fatal.
PIXEL_CALLS = frozenset({
    "read", "retrieve", "imshow", "imwrite", "imencode", "imdecode",
    "tofile", "save", "frombuffer", "namedWindow", "waitKey",
})

# The three consumer rates the claim is about. 0.0 means "as fast as the
# device will hand them over" -- the case the drain must cost nothing in.
# 6 and 3 are the cases it exists for: his configured preview_fps was 6
# when the stale-frame lag was first noticed, against 15.0 delivered.
CONSUMER_FPS = (0.0, 6.0, 3.0)
# ...and the consumer that is slow because of its OWN WORK rather than a
# sleep it can shorten. A paced arm hides what the drain costs -- it sleeps
# for whatever is left of the period, so the drain's time comes out of the
# sleep -- and a consumer doing 80 ms of detection per cycle has no sleep to
# take it out of. 80 ms against a 67 ms device is the marginal case: barely
# slower than the device, which is where his preview actually sits.
CONSUMER_WORK_S = (0.080, 0.300)
DEFAULT_SECONDS = 6.0


def self_check(path: str = __file__) -> list:
    """The names in this file's CODE that are on the forbidden list. Empty
    means the file may run. Strings and comments are skipped, which is why
    the docstring above can discuss the forbidden calls by name."""
    hits = []
    source = Path(path).read_bytes()
    for tok in tokenize.tokenize(io.BytesIO(source).readline):
        if tok.type == tokenize.NAME and tok.string in PIXEL_CALLS:
            hits.append("%s line %d" % (tok.string, tok.start[0]))
    return hits


def holders(node: str = "/dev/video0") -> list:
    """The pids holding ``node`` open, from /proc. Best effort: a process
    belonging to another user is invisible here, so an empty list is "none
    that I can see", not a guarantee."""
    found = []
    try:
        pids = [name for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return found
    for pid in pids:
        if pid == str(os.getpid()):
            continue
        fd_dir = "/proc/%s/fd" % pid
        try:
            names = os.listdir(fd_dir)
        except OSError:                  # gone, or not ours to look at
            continue
        for name in names:
            try:
                target = os.readlink(os.path.join(fd_dir, name))
            except OSError:
                continue
            if target == node:
                found.append(int(pid))
                break
    return sorted(set(found))


class ModelledQueue:
    """A V4L2-SHAPED device, in this process: a fixed-depth buffer queue the
    device fills on a fixed interval, and a ``grab()`` that pops the oldest
    or WAITS for the next one. Real time, real sleeps.

    It exists because the camera is often busy -- Jarvis holds /dev/video0
    whenever it is running, and the house rule is to say so and stop rather
    than take the lens off the assistant for a number. Against this, the
    drain's own arithmetic can still be measured end to end and the run is
    repeatable by anyone; what it CANNOT establish is the device's real
    delivered interval, which is why that number is supplied on the command
    line from the one thing already measured at the hardware
    (scripts/camera_mode_probe.py: 15.0 fps, 67.9 ms, 2026-09-03).

    A "buffer" here is an integer ordinal. Nothing decodes and there are no
    pixels in this file to decode.
    """

    def __init__(self, fps: float = 15.0, depth: int = 4,
                 now=time.perf_counter):
        self.interval = 1.0 / max(0.001, float(fps))
        self.depth = int(depth)
        self._now = now
        self.next_at = now() + self.interval
        self.queue = []
        self.produced = 0
        self.overrun = 0          # buffers the DRIVER threw away: queue full
        self.grabs = 0

    def _fill(self) -> None:
        while self._now() >= self.next_at:
            self.produced += 1
            self.queue.append(self.produced)
            if len(self.queue) > self.depth:
                self.queue.pop(0)
                self.overrun += 1
            self.next_at += self.interval

    def grab(self) -> bool:
        self.grabs += 1
        self._fill()
        if not self.queue:
            time.sleep(max(0.0, self.next_at - self._now()))
            self._fill()
        if not self.queue:
            return False
        # The ordinal is popped and DISCARDED. Even the model never hands a
        # buffer to anything; the drain's business is which one it stops on.
        self.queue.pop(0)
        return True

    def release(self) -> None:
        pass


def run_arm(cap_class, cap, fps: float, seconds: float, max_drops: int,
            now=time.perf_counter, depth: int = 4, nominal: float = 30.0,
            work_s: float = 0.0) -> dict:
    """One arm: run a consumer for ``seconds`` and drain each cycle.

    ``fps`` paces it -- sleeping whatever is left of the period, which
    ABSORBS the drain's own cost -- and ``fps`` of 0 means no pacing at all.
    ``work_s`` is the other kind of slow: time the consumer spends on its
    own work, which nothing absorbs, so the drain's cost lands on top of it.

    Nothing is decoded. Each cycle calls ``drain()``, which grabs and
    discards; the buffer it stops on is left in the device exactly as the
    app leaves it for its own decode.
    """
    drain = cap_class(cap, max_drops=max_drops, depth=depth,
                      nominal_fps=nominal)
    period = (1.0 / fps) if fps > 0 else 0.0
    cycles = 0
    dropped = 0
    bounded = 0
    worst_ms = 0.0
    spent_ms = 0.0
    started = now()
    deadline = started + seconds
    while now() < deadline:
        t0 = now()
        got = drain.drain()
        if got["grabbed"] <= 0:
            return {"ok": False, "cycles": cycles}
        cycles += 1
        dropped += got["dropped"]
        bounded += 1 if got["bounded"] else 0
        spent_ms += got["spent_ms"]
        worst_ms = max(worst_ms, got["spent_ms"])
        if work_s:
            time.sleep(work_s)
        if period:
            left = period - (now() - t0)
            if left > 0:
                time.sleep(left)
    elapsed = max(1e-9, now() - started)
    return {"ok": cycles > 0, "cycles": cycles, "elapsed": elapsed,
            "fps": cycles / elapsed, "dropped": dropped,
            "drop_fps": dropped / elapsed, "bounded": bounded,
            "mean_ms": spent_ms / max(1, cycles),
            "worst_ms": worst_ms, "longest_ms": drain.longest_grab_ms,
            "interval_ms": drain.interval_s * 1000.0,
            "frames_ps": (cycles + dropped) / elapsed}


def line(label: str, arm: dict) -> str:
    if not arm.get("ok"):
        return "  %-22s grab FAILED after %d cycles" % (label,
                                                        arm.get("cycles", 0))
    return ("  %-26s %5.1f fps  drop %5.1f/s  device %5.1f fps  "
            "drain %5.2f ms (worst %6.2f)  interval %5.1f ms  bounded %d" % (
                label, arm["fps"], arm["drop_fps"], arm["frames_ps"],
                arm["mean_ms"], arm["worst_ms"], arm["interval_ms"],
                arm["bounded"]))


def run_model(fps: float, seconds: float) -> int:
    """Every arm against a modelled queue instead of the camera. Says so in
    its own first line, so a pasted result cannot be mistaken for a
    measurement at the hardware."""
    from jarvis import camera as cam                # noqa: PLC0415 - lazy
    print("camera drain probe -- MODELLED DEVICE, NOT THE CAMERA. No "
          "/dev/video node is opened.")
    print("modelled at %.1f fps (%.1f ms interval), a %d-buffer queue; the "
          "rate is the one measured at his LifeCam on 2026-09-03 with "
          "scripts/camera_mode_probe.py." % (fps, 1000.0 / fps, 4))
    print("drain: queue %d deep (drops %d a read), %.0f ms budget, %.0f fps "
          "nominal seed" % (4, min(cam.DRAIN_MAX_DROPS, 3),
                            cam.DRAIN_BUDGET_S * 1000.0, 30.0))
    print("")
    print("  consumer                    delivered    discarded   "
          "device rate   cost per cycle")
    plans = [(("as fast as it will" if c <= 0 else "%.0f fps" % c), c, 0.0)
             for c in CONSUMER_FPS]
    plans += [("%.0f ms of work" % (w * 1000.0), 0.0, w)
              for w in CONSUMER_WORK_S]
    for name, consumer, work in plans:
        for label, drops in (("drained", cam.DRAIN_MAX_DROPS),
                             ("undrained", 0)):
            model = ModelledQueue(fps)
            arm = run_arm(cam.DrainingCapture, model, consumer, seconds,
                          drops, depth=4, nominal=30.0, work_s=work)
            arm["overrun"] = model.overrun
            print(line("%s, %s" % (name, label), arm)
                  + "  driver lost %d" % model.overrun)
    print("")
    print("'driver lost' is the buffers the DEVICE overwrote because the")
    print("queue was full -- the staleness the undrained arm is reading.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="0",
                    help="index or /dev node; the app opens 0 (default 0)")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                    help="seconds per arm (default %.0f)" % DEFAULT_SECONDS)
    ap.add_argument("--node", default="/dev/video0",
                    help="the node checked for another holder")
    ap.add_argument("--force-busy", action="store_true",
                    help="skip the check for another process holding the "
                         "node. The device will still refuse to stream for "
                         "a second reader, so this buys an error message "
                         "rather than a measurement -- use --model instead")
    ap.add_argument("--model", type=float, default=0.0, metavar="FPS",
                    help="do not open the camera at all -- run the arms "
                         "against a modelled V4L2 queue delivering FPS "
                         "frames a second (his LifeCam measures 15.0). Use "
                         "this when the camera is busy; the drain's own "
                         "arithmetic is real, the device is not.")
    args = ap.parse_args(argv)

    hits = self_check()
    if hits:
        print("REFUSING TO RUN: this file's code names a pixel-bearing "
              "call: %s" % ", ".join(hits))
        return 4

    if args.model > 0:
        return run_model(args.model, args.seconds)

    if not args.force_busy:
        busy = holders(args.node)
        if busy:
            print("STOPPED: %s is held by pid %s -- that is almost certainly "
                  "the running Jarvis. Not taking the camera off it. Stop "
                  "Jarvis first (the pid file is /tmp/vss_voice/jarvis.pid) "
                  "and run this again." % (args.node,
                                           ", ".join(str(p) for p in busy)))
            return 3

    try:
        from jarvis import camera as cam            # noqa: PLC0415 - lazy
    except Exception as exc:  # noqa: BLE001
        print("STOPPED: jarvis.camera is not importable (%s)" % exc)
        return 3
    try:
        import cv2                                  # noqa: PLC0415 - lazy
    except Exception as exc:  # noqa: BLE001
        print("STOPPED: cv2 is not importable (%s)" % exc)
        return 3

    target = args.device
    if isinstance(target, str) and target.isdigit():
        target = int(target)
    elif isinstance(target, str) and not Path(target).exists():
        print("STOPPED: %s does not exist" % target)
        return 3

    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        cap.release()
        print("STOPPED: could not open device %r -- busy or absent. Not "
              "retrying." % (target,))
        return 3
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        got = cam.capture_mode(cap)
        print("camera drain probe -- numbers only. Buffers are dequeued "
              "with grab() and never decoded, shown or saved.")
        print("cv2 %s  device %r  granted %.0fx%.0f %s at %.1f fps nominal, "
              "%.0f driver buffer(s)" % (
                  cv2.__version__, target, got["width"], got["height"],
                  got["fourcc"] or "?", got["fps"], got["buffersize"]))
        depth = int(got["buffersize"]) if got["buffersize"] >= 1 else 4
        nominal = float(got["fps"]) if got["fps"] > 0 else 30.0
        print("drain: queue %d deep (drops %d a read), %.0f ms budget, "
              "%.0f fps nominal seed" % (
                  depth, min(cam.DRAIN_MAX_DROPS, depth - 1),
                  cam.DRAIN_BUDGET_S * 1000.0, nominal))
        print("")
        print("  consumer                    delivered    discarded   "
              "device rate   cost per cycle")
        failed = False
        plans = [(("as fast as it will" if c <= 0 else "%.0f fps" % c), c,
                  0.0) for c in CONSUMER_FPS]
        plans += [("%.0f ms of work" % (w * 1000.0), 0.0, w)
                  for w in CONSUMER_WORK_S]
        for name, consumer, work in plans:
            for label, drops in (("drained", cam.DRAIN_MAX_DROPS),
                                 ("undrained", 0)):
                arm = run_arm(cam.DrainingCapture, cap, consumer,
                              args.seconds, drops, depth=depth,
                              nominal=nominal, work_s=work)
                print(line("%s, %s" % (name, label), arm))
                if not arm.get("ok"):
                    failed = True
                    break
            if failed:
                break
    finally:
        cap.release()
    if failed:
        print("STOPPED: the device would not grab. Not retrying.")
        return 1
    for note in (
            "Compare the DRAINED row with the UNDRAINED one at the same",
            "consumer rate. 'delivered' must MATCH between them: the drain",
            "may not cost frame rate. 'drop/s' is what it threw away to",
            "reach the newest frame -- 0 on the undrained row by",
            "construction, where those frames are the staleness instead.",
            "'device' is delivered + dropped, the device's own rate: when",
            "the drained row reaches it and the undrained one does not,",
            "the difference is the queue backing up behind a slow reader.",
            "'interval' is the delivered frame interval the drain",
            "measured for itself, which is what its arithmetic divides by.",
            "'bounded' counts cycles stopped by a limit -- the queue's own",
            "depth, the max_drops backstop, or the time budget -- with",
            "buffers the ledger still counted; a whole column of them means",
            "the consumer is further behind than one read can catch up."):
        print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
