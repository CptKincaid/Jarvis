#!/usr/bin/env python3
"""The grab-and-throw gesture on HIS hand, in numbers only.

    ~/vss_env/bin/python scripts/gesture_selfcheck.py --synthetic
    ~/vss_env/bin/python scripts/gesture_selfcheck.py --seconds 30
    ~/vss_env/bin/python scripts/gesture_selfcheck.py --seconds 30 --json

WHAT HE DOES WITH IT. Runs it, sits at the desk, reaches at the lens with
an open hand, closes it, holds still, flings it left or right, and reads
the numbers back. Every threshold in jarvis/gesture.py is a calibrated
STARTING POINT taken from a synthetic hand and his lens constants; this is
the only instrument that turns any of them into a measurement, and nobody
has to look at a frame to use it. It prints, in order:

 1. what the sensing owner says -- offline mode, the curfew. If the camera
    is not allowed NOTHING below runs and the device is not opened. That is
    the same rule the running Jarvis obeys, reached through the same object;
 2. the hand models: on disk, sizes against the pinned ones, licences,
    whether the SHA256SUMS and LICENSE texts beside them agree with the
    code, and whether ONNX Runtime can build them. A missing model STOPS
    the run with the reason -- it never becomes "no hand seen";
 3. the face detector, because the reach ratio R = palm_diag / interocular
    needs a face in the SAME frame. Without one R is 0.0 by construction
    and nothing can grab, and the report says so rather than blaming his
    hand;
 4. the thresholds, and what the frame counters mean in time at the capture
    rate the config asks for;
 5. one line per frame -- faces, interocular px, hands, palm_diag, C (the
    closed scalar), R (the reach ratio), the centroid, the machine's state,
    dwell and displacement, and any event -- then a summary: the hand hit
    rate, R and C min/p50/max, the 3-D closed scalar from the network's
    world landmarks (logged, not used), per-stage milliseconds, every event
    with its sector and why, and a pass/fail line per claim.

HOW TO READ IT. If "reached" fails, the largest R is printed beside the bar:
come closer, or present the knuckles to the lens (R is pose-dependent and
the design's margin is thin edge-on). If "open seen" or "closed seen"
fails, the C bars 0.85 / 0.70 did not fit his hand and the printed C range
is the number to tune from. If a grab fires and the throw does not, the
dist_u column across the swing is the displacement the fling bars are
judged on.

NO IMAGE DATA LEAVES THIS SCRIPT. It prints no pixels, saves no frame, opens
no window, and every number it prints is a count, a size, a ratio, an angle
or a millisecond. ``jarvis/visionrig.assert_numbers_only`` is run over the
report before it is printed, so this is enforced rather than promised.
Hunter can run it against his real camera and paste the output to anyone,
including me, without either of us ever seeing his room.

``--synthetic`` draws an open hand-shaped blob and runs the same loop with no
lens and no device. It measures COST and PLUMBING only: a drawn hand is not
a hand, and there is no face in the frame, so R is 0.0 and nothing can
grab. It is what the suite runs.

Exit codes: 0 every measurable claim held, 1 one did not, 2 sensing said
no (offline or curfew), 3 there is no camera to check.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402

from jarvis import camera as cam                          # noqa: E402
from jarvis import facedetect                             # noqa: E402
from jarvis import gesture as g                           # noqa: E402
from jarvis import handpose as hp                         # noqa: E402
from jarvis import visionrig as vr                        # noqa: E402
from jarvis.assistant_config import AssistantConfig       # noqa: E402
from jarvis.sensing import SensingPolicy                   # noqa: E402

BANNER = (
    "gesture self-check -- numbers only. No frame is displayed, saved or "
    "described;\nevery line below is a count, a size, a ratio, an angle or "
    "a millisecond.")

# A face seen within this long may lend its interocular distance to a frame
# the reaching arm has just covered it in (design: "a face seen within the
# last 5 s ... may be reused as a stale median; older than that is a miss").
FACE_STALE_S = 5.0


class ConfigOverlay:
    """His config, read-only, with command-line overrides laid on top.

    Never ``AssistantConfig.set()``: that SAVES, and a self-check must not
    write his assistant.json. Everything downstream (jarvis/camera,
    jarvis/sensing, this script) reads through ``get(dotted, default)``,
    so an overlay is all an override needs.
    """

    def __init__(self, cfg, overrides: dict):
        self._cfg = cfg
        self._over = dict(overrides)

    def get(self, key: str, default=None):
        if key in self._over:
            return self._over[key]
        return self._cfg.get(key, default)


# ----------------------------------------------------------- thresholds
def thresholds_from_config(cfg) -> g.CastThresholds:
    """``camera.gesture.<field>`` overrides on top of the design defaults.

    The wiring branch owns the config block; this reads whatever of it
    exists so the self-check measures the same machine Jarvis would run,
    and falls back to the design numbers key by key when it does not.
    """
    base = g.CastThresholds()
    over = {}
    for field in dataclasses.fields(base):
        raw = cfg.get("camera.gesture." + field.name, None)
        if raw is None:
            continue
        default = getattr(base, field.name)
        if isinstance(default, tuple):
            over[field.name] = tuple(str(s) for s in raw)
        elif isinstance(default, bool):
            over[field.name] = bool(raw)
        elif isinstance(default, int):
            over[field.name] = int(raw)
        else:
            over[field.name] = float(raw)
    return dataclasses.replace(base, **over) if over else base


def threshold_lines(t: g.CastThresholds, fps: float) -> list:
    ms = 1000.0 / max(float(fps), 0.1)
    return [
        g.describe(t, fps),
        "closed C <= %.2f, open C >= %.2f (dead band between; MEASURED on a "
        "synthetic hand, fist max 0.680 / open min 0.919)"
        % (t.closed_max, t.open_min),
        "reach R >= %.2f to grab, >= %.2f to arm (MEASURED: a hand at the "
        "face plane never exceeded 2.03; anthropometry behind R is GUESSED)"
        % (t.reach_min, t.reach_arm),
        "throw bars in hand-units: released in frame %.2f, left near an "
        "edge %.2f, lost mid-air %.2f with last step %.2f"
        % (t.throw_release_u, t.throw_exit_u, t.throw_lost_u,
           t.exit_step_u),
        "dwell %d frames = %.0f ms, still within %.2f units; grace %d; "
        "cooldown %d; carry cap %d frames / %.1f s"
        % (t.dwell_frames, t.dwell_frames * ms, t.anchor_drift_u,
           t.lost_grace_frames, t.cooldown_frames, t.carry_max_frames,
           t.carry_max_s),
    ]


# ------------------------------------------------------ synthetic frames
def draw_hand(width: int, height: int, cx: float, cy: float,
              scale: float, rng) -> np.ndarray:
    """An open hand-shaped blob: a palm ellipse, four fingers up, a thumb
    out. The frames lane measured this shape firing the real palm
    detector; it is not a hand and says nothing about his."""
    import cv2                            # noqa: PLC0415 - synthetic only
    img = np.full((height, width, 3), (40, 45, 55), np.uint8)
    img = (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
    skin = (150, 178, 214)
    s = 100.0 * scale
    cv2.ellipse(img, (int(cx), int(cy)), (int(.42 * s), int(.55 * s)),
                0, 0, 360, skin, -1)
    for ang, ln, wd in ((-78, 1.00, .15), (-90, 1.10, .15),
                        (-102, 1.02, .15), (-114, .85, .13),
                        (-160, .70, .17)):
        a = math.radians(ang)
        bx, by = cx + .30 * s * math.cos(a), cy + .30 * s * math.sin(a)
        tx, ty = cx + ln * s * math.cos(a), cy + ln * s * math.sin(a)
        cv2.line(img, (int(bx), int(by)), (int(tx), int(ty)), skin,
                 int(wd * s))
        cv2.circle(img, (int(tx), int(ty)), int(wd * s / 2), skin, -1)
    cv2.ellipse(img, (int(cx), int(cy + .18 * s)),
                (int(.40 * s), int(.34 * s)), 0, 0, 360, skin, -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


class SyntheticHandSource:
    """Frames this process draws: an open hand drifting across the middle
    of the picture. No device, no lens. COST and PLUMBING only."""

    def __init__(self, width: int = 1280, height: int = 720, frames: int = 0,
                 seed: int = 7):
        self.width, self.height = int(width), int(height)
        self.left = int(frames)
        self.reads = 0
        self.released = 0
        self._rng = np.random.default_rng(seed)

    def read(self):
        if self.left and self.reads >= self.left:
            return False, None
        self.reads += 1
        k = self.reads
        cx = self.width / 2.0 + 0.12 * self.width * math.sin(k / 9.0)
        cy = self.height / 2.0 + 0.05 * self.height * math.cos(k / 7.0)
        return True, draw_hand(self.width, self.height, cx, cy,
                               2.0 * self.width / 1280.0, self._rng)

    def release(self) -> None:
        self.released += 1


# ------------------------------------------------------------- the loop
def _pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[min(len(s) - 1, int(round(q * (len(s) - 1))))])


def _stats(values) -> dict:
    return {"n": len(values), "min": round(min(values), 4) if values else 0.0,
            "p50": round(_pct(values, 0.5), 4),
            "max": round(max(values), 4) if values else 0.0}


def closed_3d(world) -> float:
    """The closed scalar on the network's metric 3-D landmarks. LOGGED, NOT
    USED: in simulation it is pose-invariant (gap 0.821 against the 2-D
    scalar's 0.239) but that was synthetic 3-D, and only his run says
    whether the network's is trustworthy."""
    w = np.asarray(world, dtype=np.float64)
    if w.shape != (21, 3):
        return 0.0
    palm_len = float(np.linalg.norm(w[g.MID_MCP] - w[g.WRIST]))
    palm_w = float(np.linalg.norm(w[g.PNK_MCP] - w[g.IDX_MCP]))
    diag = math.hypot(palm_len, palm_w)
    if diag < 1e-9:
        return 0.0
    span = float(np.mean([np.linalg.norm(w[t] - w[g.WRIST])
                          for t in g.TIPS]))
    return span / diag


def run_loop(source, detector, lens, head, tracker, machine, *,
             min_conf: float, frames: int, seconds, every: int, say,
             now=time.monotonic) -> dict:
    """Frames in, one line and a handful of numbers out, per frame. The
    frame is a local of this loop and nothing about it survives the
    iteration: the detector, the tracker and the machine all return
    scalars."""
    dw, dh = getattr(detector, "input_size", (0, 0)) if detector else (0, 0)
    sx = lens.width_px / float(dw) if dw else 1.0
    sy = lens.height_px / float(dh) if dh else 1.0
    recent_eyes = collections.deque()          # (t, eye_px) within 5 s
    acc = {"grab_ms": [], "face_ms": [], "hand_ms": [], "R": [], "C": [],
           "C3": [], "diag": [], "conf": [], "handed": []}
    events = []
    states = collections.Counter()
    read = hand_frames = face_frames = stale_frames = 0
    start = now()
    say("   %4s %6s | %5s %6s | %5s %6s %6s %5s %5s %9s | %-9s %3s %5s | %s"
        % ("f", "ms", "faces", "eye", "hands", "diag", "C", "R", "C3",
           "centroid", "state", "dw", "dist", "event"))
    for i in range(int(frames)):
        if seconds is not None and now() - start >= seconds:
            break
        t0 = now()
        ok, frame = source.read()
        t1 = now()
        if not ok or frame is None:
            break
        read += 1
        acc["grab_ms"].append((t1 - t0) * 1000.0)

        faces = 0
        eye_px = 0.0
        stale = False
        if detector is not None:
            t2 = now()
            rows = detector.detect(frame)
            acc["face_ms"].append((now() - t2) * 1000.0)
            obs = [vr.observe(r, lens, sx, sy, head)
                   for r in (rows if rows is not None else [])]
            faces = len(obs)
            best = max(obs, key=lambda o: o.conf) if obs else None
            if (best is not None and best.landmarks_ok
                    and best.conf >= min_conf and best.eye_px > 0.0):
                recent_eyes.append((t1, float(best.eye_px)))
                face_frames += 1
            while recent_eyes and t1 - recent_eyes[0][0] > FACE_STALE_S:
                recent_eyes.popleft()
            if recent_eyes:
                eye_px = statistics.median(e for _t, e in recent_eyes)
                stale = not (best is not None and best.conf >= min_conf)
                stale_frames += int(stale)

        t3 = now()
        rows = tracker.detect(frame)
        acc["hand_ms"].append((now() - t3) * 1000.0)
        hands = tuple(g.observe_hand(r.lm, eye_px, r.conf) for r in rows)
        event = machine.update(hands, eye_px, i)
        del frame                          # the last reference, explicitly

        top = hands[0] if hands else None
        if top is not None and top.ok:
            hand_frames += 1
            acc["R"].append(top.reach)
            acc["C"].append(top.closed)
            acc["diag"].append(top.palm_diag)
            acc["conf"].append(top.conf)
            acc["handed"].append(float(rows[0].handed))
            c3 = closed_3d(rows[0].world)
            acc["C3"].append(c3)
        else:
            c3 = 0.0
        st = machine.status()
        states[st["state"]] += 1
        if event is not None:
            events.append(event.numbers_only())
        if every > 0 and (i % every == 0 or event is not None):
            say("   %4d %6.0f | %5d %6.1f%s | %5d %6.1f %6.3f %5.2f %5.2f "
                "(%4.0f,%4.0f) | %-9s %3d %5.2f | %s"
                % (i, (now() - start) * 1000.0, faces, eye_px,
                   "*" if stale else " ", len(hands),
                   top.palm_diag if top else 0.0,
                   top.closed if top else 0.0, top.reach if top else 0.0,
                   c3, top.cx if top else 0.0, top.cy if top else 0.0,
                   st["state"], st["dwell"], st["dist_u"],
                   "%s %s %s" % (event.kind.upper(), event.sector,
                                 event.why) if event else ""))
    elapsed = now() - start
    return {
        "frames": read, "seconds": round(elapsed, 3),
        "fps": round(read / elapsed, 2) if elapsed > 0 else 0.0,
        "face_frames": face_frames, "stale_face_frames": stale_frames,
        "hand_frames": hand_frames,
        "hand_hit_rate": round(hand_frames / read, 4) if read else 0.0,
        "R": _stats(acc["R"]), "C": _stats(acc["C"]), "C3": _stats(acc["C3"]),
        "palm_diag": _stats(acc["diag"]), "conf": _stats(acc["conf"]),
        "handed_mean": round(statistics.fmean(acc["handed"]), 4)
        if acc["handed"] else 0.0,
        "ms": {k: {"p50": round(_pct(v, 0.5), 2),
                   "p95": round(_pct(v, 0.95), 2)}
               for k, v in (("grab", acc["grab_ms"]), ("face", acc["face_ms"]),
                            ("hand", acc["hand_ms"]))},
        "states": dict(states),
        "events": events,
        "grabs": sum(e["kind"] == "grab" for e in events),
        "throws": sum(e["kind"] == "throw" for e in events),
        "drops": sum(e["kind"] == "drop" for e in events),
        "final": machine.status(),
    }


def checks(run: dict, t: g.CastThresholds, *, synthetic: bool,
           detector_ok: bool) -> list:
    """(name, passed, detail, measurable). The synthetic run can only speak
    to plumbing and cost; the rest is his to measure."""
    out = [("hand seen", run["hand_frames"] >= 1,
            "%d of %d frames (hit rate %.2f)"
            % (run["hand_frames"], run["frames"], run["hand_hit_rate"]),
            True),
           ("hand stage cost", run["ms"]["hand"]["p50"] <= 40.0,
            "p50 %.1f ms, p95 %.1f ms at 2 threads (design 9.0)"
            % (run["ms"]["hand"]["p50"], run["ms"]["hand"]["p95"]), True)]
    live = not synthetic
    out.append(("face seen", run["face_frames"] >= 1,
                "%d frames with a face above min_conf" % run["face_frames"]
                if detector_ok else "no face detector: R is 0.0 without one",
                live))
    out.append(("open seen", run["C"]["max"] >= t.open_min,
                "C max %.3f against open bar %.2f"
                % (run["C"]["max"], t.open_min), live))
    out.append(("closed seen",
                run["C"]["n"] > 0 and run["C"]["min"] <= t.closed_max,
                "C min %.3f against closed bar %.2f"
                % (run["C"]["min"], t.closed_max), live))
    out.append(("reached", run["R"]["max"] >= t.reach_min,
                "R max %.2f against grab bar %.2f (arm %.2f)"
                % (run["R"]["max"], t.reach_min, t.reach_arm), live))
    out.append(("grab fired", run["grabs"] >= 1, "%d grabs" % run["grabs"],
                live))
    out.append(("throw fired", run["throws"] >= 1,
                "%d throws, sectors %s" % (
                    run["throws"],
                    ",".join(e["sector"] for e in run["events"]
                             if e["kind"] == "throw") or "-"), live))
    return out


# --------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=240,
                    help="frames to run (default 240, ~40 s at 6 fps)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long instead of after --frames")
    ap.add_argument("--every", type=int, default=1,
                    help="print every Nth frame line (events always print)")
    ap.add_argument("--device", default=None,
                    help="override camera.device, e.g. /dev/video2")
    ap.add_argument("--deep", action="store_true",
                    help="hash the model files as well as sizing them")
    ap.add_argument("--models-only", action="store_true",
                    help="check the weights and stop; opens no device")
    ap.add_argument("--mirrored", action="store_true",
                    help="the feed is mirrored (camera.mirrored); flips "
                         "only which side is called left")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on frames this process draws instead of a "
                         "camera. Opens no device. Measures COST and "
                         "PLUMBING, never accuracy: a drawn hand is not a "
                         "hand and it has no face, so R is 0.0")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable, and just as pixel-free")
    args = ap.parse_args(argv)

    report: dict = {"banner": BANNER}
    say = (lambda *a: None) if args.json else print
    say(BANNER)
    say("")

    cfg = AssistantConfig.load()
    if args.device is not None:
        cfg = ConfigOverlay(cfg, {"camera.device": args.device})

    # ------------------------------------------------------------ 1. sensing
    policy = SensingPolicy(cfg=cfg)
    status = policy.status()
    report["sensing"] = status
    say("1. sensing")
    say("   camera=%s offline=%s reason=%s curfew=%s"
        % (status["camera"], status["offline"], status["reason"] or "-",
           status.get("curfew", "") or "-"))

    # ------------------------------------------------------- 2. hand models
    model_dir = str(cfg.get("camera.gesture.model_dir", "") or "") or None
    probe = hp.probe(model_dir=model_dir, deep=args.deep)
    report["hand_models"] = probe
    say("")
    say("2. hand models  (%s)" % probe["dir"])
    for key in sorted(probe["models"]):
        m = probe["models"][key]
        say("   %-14s %-5s %9d B (want %9d)  %-10s licence %s  %s"
            % (key, "ok" if m["ok"] else "MISS", m["bytes"],
               m["expected_bytes"], m["licence"],
               "matches" if m["licence_matches"] else "MISSING/DIFFERENT",
               m["reason"]))
    prov = probe["provenance"]
    say("   provenance: SHA256SUMS %s, %d entries, agrees with the code: %s"
        % ("present" if prov["sums_present"] else "MISSING",
           prov["entries"], prov["ok"]))
    say("   ort %s (%s)  cv2 %s  threads %d  detect %s  anchors %d"
        % (probe["ort"]["version"] or "-", probe["ort"]["providers"] or "-",
           probe["cv2"]["version"] or "-", probe["threads"],
           probe["detect_size"], probe["anchors"]))
    if args.models_only:
        return _finish(report, args, 0 if probe["ready"] else 1)
    if not probe["ready"]:
        say("")
        say("STOPPED: a hand model is unusable. Nothing falls back; see the "
            "reason above.")
        return _finish(report, args, 1)

    # ----------------------------------------------------- 3. face detector
    detector, det_why = cam.detector_from_config(
        cfg, score_threshold=facedetect.PROBE_THRESHOLD)
    report["face_detector"] = {"ok": detector is not None,
                               "reason": det_why}
    say("")
    say("3. face detector  (R needs a face in the same frame)")
    say("   %s" % ("ready" if detector is not None
                   else "ABSENT: %s -- R will be 0.0 and nothing can grab"
                   % det_why))

    # -------------------------------------------------------- 4. thresholds
    fps = float(cfg.get("camera.preview_fps", 6.0))
    configured = thresholds_from_config(cfg)
    # THE SAME SCALING THE APP MUST APPLY. Every counter in the machine is
    # in frames and was designed for 5.5-8 fps; above that band for_fps
    # scales them up, so a config asking for 15 fps gets a 6-frame dwell
    # rather than a 200 ms one. The catch, and the reason section 5 prints
    # the DELIVERED rate beside this one: the LifeCam delivers ~7.5 fps
    # whatever preview_fps asks for, so scaling by the asked-for rate on a
    # camera that cannot deliver it doubles the felt dwell. The warning at
    # the end of the run is what says so, from a measurement.
    thresholds = g.CastThresholds.for_fps(fps, configured)
    report["thresholds"] = dataclasses.asdict(thresholds)
    report["thresholds"]["target_sectors"] = list(thresholds.target_sectors)
    report["capture_fps"] = fps
    report["counters_scaled"] = thresholds != configured
    say("")
    say("4. thresholds at camera.preview_fps %.1f%s"
        % (fps, "  (frame counters SCALED UP from the 7.5 fps design "
                "by for_fps; see the fps warning after the run)"
           if thresholds != configured else ""))
    for line in threshold_lines(thresholds, fps):
        say("   " + line)

    try:
        tracker = hp.HandTracker(
            model_dir=model_dir,
            threads=int(cfg.get("camera.gesture.threads",
                                cfg.get("camera.threads", 2))),
            deep=args.deep)
    except hp.HandModelUnavailable as exc:
        report["tracker_error"] = str(exc)
        say("")
        say("STOPPED: %s" % exc)
        return _finish(report, args, 1)
    mirrored = bool(args.mirrored or cfg.get("camera.mirrored", False))
    events_seen = []
    head = cam.head_from_config(cfg)
    min_conf = float(cfg.get("camera.min_conf", 0.6))

    # -------------------------------------------------------- 5. the run
    if args.synthetic:
        try:
            lens = cam.lens_from_config(cfg)
        except ValueError:
            lens = cam.Lens(width_px=1280, height_px=720, hfov_deg=65.6)
        machine = g.CastGesture(thresholds, (lens.width_px, lens.height_px),
                                mirrored=mirrored, on_event=events_seen.append,
                                preview_fps=fps)
        say("")
        say("5. the loop on SYNTHETIC frames  (%d frames, no device)"
            % args.frames)
        say("   A drawn hand is not a hand and there is no face, so R is 0.0 "
            "by construction and nothing can grab. Cost and plumbing only.")
        source = SyntheticHandSource(lens.width_px, lens.height_px,
                                     frames=args.frames)
        run = run_loop(source, detector, lens, head, tracker, machine,
                       min_conf=min_conf, frames=args.frames,
                       seconds=args.seconds, every=args.every, say=say)
        source.release()
        return _report_run(report, args, run, thresholds, say,
                           synthetic=True, detector_ok=detector is not None)

    if not status["camera"]:
        say("")
        say("STOPPED: sensing says the camera may not run (%s). The device "
            "was not opened." % (status["reason"] or "denied"))
        return _finish(report, args, 2)
    if not cam.device_nodes():
        say("")
        say("STOPPED: no /dev/video* on this box.")
        return _finish(report, args, 3)

    feed, why = cam.build(cfg, policy)
    if feed is None:
        # camera.enabled is False on a fresh config, and that must not stop
        # a bring-up check -- it is the thing he runs BEFORE turning it on.
        try:
            lens = cam.lens_from_config(cfg)
            device = str(cfg.get("camera.device", "") or "")
            feed = cam.CameraFeed(
                policy,
                lambda: cam.open_capture(device, lens.width_px,
                                         lens.height_px,
                                         str(cfg.get("camera.fourcc",
                                                     "MJPG") or "")),
                lens=lens, present=lambda: cam.device_present(device))
            say("")
            say("   (%s -- checking anyway, that is what this script is for)"
                % why)
        except Exception as exc:  # noqa: BLE001
            report["build_error"] = "%s: %s" % (type(exc).__name__, exc)
            say("")
            say("STOPPED: %s" % report["build_error"])
            return _finish(report, args, 1)
    lens = feed.lens
    machine = g.CastGesture(thresholds, (lens.width_px, lens.height_px),
                            mirrored=mirrored, on_event=events_seen.append,
                            preview_fps=fps)
    say("")
    say("5. the loop on the camera  (%s, mirrored=%s)"
        % ("%d frames" % args.frames if args.seconds is None
           else "%.0f s" % args.seconds, mirrored))
    say("   Reach at the lens with an open hand, close it, hold still, "
        "fling it left or right. Open it where it is to put it down.")
    source = cam.FeedSource(feed)
    try:
        run = run_loop(source, detector, lens, head, tracker, machine,
                       min_conf=min_conf, frames=args.frames,
                       seconds=args.seconds, every=args.every, say=say)
    finally:
        feed.close()
    return _report_run(report, args, run, thresholds, say,
                       synthetic=False, detector_ok=detector is not None)


def _report_run(report, args, run, thresholds, say, *, synthetic,
                detector_ok) -> int:
    report["run"] = run
    say("")
    say("   frames %d in %.1f s (%.1f fps)  faces %d (%d stale)  hands %d "
        "(hit rate %.2f)"
        % (run["frames"], run["seconds"], run["fps"], run["face_frames"],
           run["stale_face_frames"], run["hand_frames"],
           run["hand_hit_rate"]))
    for key in ("R", "C", "C3", "palm_diag", "conf"):
        s = run[key]
        say("   %-9s n %4d  min %8.3f  p50 %8.3f  max %8.3f"
            % (key, s["n"], s["min"], s["p50"], s["max"]))
    say("   handed mean %.3f (logged, never gated on)" % run["handed_mean"])
    say("   ms p50/p95: grab %.1f/%.1f  face %.1f/%.1f  hand %.1f/%.1f"
        % (run["ms"]["grab"]["p50"], run["ms"]["grab"]["p95"],
           run["ms"]["face"]["p50"], run["ms"]["face"]["p95"],
           run["ms"]["hand"]["p50"], run["ms"]["hand"]["p95"]))
    say("   states: %s" % ", ".join("%s %d" % kv
                                     for kv in sorted(run["states"].items())))
    say("   events: grabs %d  throws %d  drops %d"
        % (run["grabs"], run["throws"], run["drops"]))
    for e in run["events"]:
        say("      f%-4d %-5s sector=%-5s toward=%-5s dist %.2f u  %s"
            % (e["frame"], e["kind"], e["sector"] or "-", e["toward"] or "-",
               e["dist_u"], e["why"]))
    asked = float(report.get("capture_fps", 0.0))
    got = float(run["fps"])
    fps_ok = (not asked or not got or synthetic
              or abs(got - asked) / asked <= 0.2)
    report["fps_mismatch"] = not fps_ok
    if not fps_ok:
        say("")
        say("   WARNING: camera.preview_fps asks for %.1f but the camera "
            "delivered %.1f fps. The frame counters were scaled for %.1f "
            "(dwell %d frames = %.0f ms at %.1f, %.0f ms at %.1f). Set "
            "camera.preview_fps to what the camera actually delivers, or "
            "the dwell is not the one that was designed."
            % (asked, got, asked, thresholds.dwell_frames,
               1000.0 * thresholds.dwell_frames / asked, asked,
               1000.0 * thresholds.dwell_frames / got, got))
    rows = checks(run, thresholds, synthetic=synthetic,
                  detector_ok=detector_ok)
    rows.append(("fps as configured", fps_ok,
                 "asked %.1f, delivered %.1f" % (asked, got),
                 not synthetic))
    report["checks"] = [{"name": n, "ok": bool(ok), "detail": d,
                         "measurable": bool(m)} for n, ok, d, m in rows]
    say("")
    say("   checks")
    failed = 0
    for name, ok, detail, measurable in rows:
        if not measurable:
            verdict = "n/a  "
        elif ok:
            verdict = "PASS "
        else:
            verdict = "FAIL "
            failed += 1
        say("   %s %-16s %s" % (verdict, name, detail))
    if synthetic:
        say("   (n/a = not measurable on a drawn hand; run it on the camera)")
    return _finish(report, args, 1 if failed else 0)


def _finish(report: dict, args, code: int) -> int:
    report["exit_code"] = code
    vr.assert_numbers_only(report)          # the promise, enforced
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
