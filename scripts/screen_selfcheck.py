#!/usr/bin/env python3
"""Can Jarvis tell WHICH SCREEN he grabbed at? Numbers only.

    ~/vss_env/bin/python scripts/screen_selfcheck.py --synthetic
    ~/vss_env/bin/python scripts/screen_selfcheck.py --minutes 10
    ~/vss_env/bin/python scripts/screen_selfcheck.py --minutes 10 --json

WHAT THIS IS FOR. Two numbers decide whether "grab a screen and throw it at
another" can work, and NEITHER HAS EVER BEEN MEASURED, on him or on anyone,
in this project:

 1. HOW ACCURATELY HE PUTS HIS HAND where he means to. I was handed
    ``anchor_drift_u = 0.60`` as "his measured placement error". It is not:
    gesture.py documents it as the DWELL STILLNESS radius, justified as
    roughly 7x the landmark noise floor. Every separation figure in the
    design is arithmetic sitting on a constant measured for a different
    purpose, and this script is what turns it into a measurement.
 2. HOW MUCH OF A GAZE SHIFT HE TAKES WITH HIS HEAD rather than his eyes.
    The screens are 41-60 degrees apart in his gaze, which is 5-13 sigma of
    head yaw IF he turns his head, and 1.4 sigma if he barely does. That is
    the difference between yaw being the stronger axis and being useless.

WHAT HE DOES. Runs it, sits where he normally sits, and works across the
three screens for ten minutes. When he reaches at a screen and closes his
hand, say which one out loud -- no, do not: say nothing at all. Just press
1 for the right-hand screen (the Spark), 2 for the middle and 3 for the
left BEFORE each grab, or run it with --label right/middle/left for one
screen at a time, which is easier and is what I would do:

    ...screen_selfcheck.py --minutes 3 --label right      # sit at the Spark
    ...screen_selfcheck.py --minutes 3 --label middle
    ...screen_selfcheck.py --minutes 3 --label left

It prints, per label: n, yaw_t at p10/p50/p90, the pooled within-label
spread, the separation between each pair on BOTH axes, the correlation
between the two, the 2-D separation, and yaw_miss_pct.

THE PASS BAR, WRITTEN DOWN IN ADVANCE so it cannot be moved afterwards:

 * Yaw is a genuine SECOND axis if d_2D >= 1.15 * max(d_hand, d_yaw) AND
   |rho| < 0.6 AND yaw_miss_pct < 20%.
 * If d_2D is within 5% of d_yaw and d_yaw is much larger than d_hand, yaw
   is not a second axis -- it is the ONLY axis, and the design should
   invert: yaw decides and the hand vetoes. jarvis/screens.py already reads
   whichever margin is stronger, so that answer needs no rewrite.
 * If |rho| > 0.6 the head and the hand are carrying one fact between them
   and the second axis is an illusion. Ship the stronger one alone.
 * Refuse to conclude on fewer than 25 grabs per label.

THE NUMBER TO LOOK AT FIRST IS yaw_miss_pct, ahead of every sigma.
handstage.py's own docstring says the reaching arm crosses the face at
exactly the moment the gesture matters, which is why attention is LATCHED
for 3 s rather than sampled. If a clean pre-reach face row is missing on a
quarter of grabs, yaw is a veto that abstains often and nothing more,
whatever the clusters look like at rest.

NO IMAGE DATA LEAVES THIS SCRIPT. It prints no pixels, saves no frame,
opens no window, and every number it prints is a count, a ratio, an angle
or a millisecond. ``jarvis.visionrig.assert_numbers_only`` is run over the
report before it is printed, so that is enforced rather than promised. He
can paste the output to anyone, including me, without either of us ever
seeing his room.

``--synthetic`` runs the same arithmetic over rows this process invents. It
measures the MATHS and the plumbing, never him, and it is what the suite
runs.

Exit codes: 0 the pass bar was reached and reported, 1 not enough samples
to conclude, 2 sensing said no (offline or curfew), 3 no camera.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jarvis import camera as cam                          # noqa: E402
from jarvis import facedetect                             # noqa: E402
from jarvis import gesture as g                           # noqa: E402
from jarvis import handpose as hp                         # noqa: E402
from jarvis import screens as sc                          # noqa: E402
from jarvis import visionrig as vr                        # noqa: E402
from jarvis.assistant_config import AssistantConfig       # noqa: E402
from jarvis.sensing import SensingPolicy                  # noqa: E402

BANNER = (
    "screen self-check -- numbers only. No frame is displayed, saved or "
    "described;\nevery line below is a count, a ratio, an angle or a "
    "millisecond.")

LABELS = ("right", "middle", "left")
MACHINE_OF = {"right": sc.SPARK, "middle": sc.HPCOMPUTER,
              "left": sc.HPCOMPUTER}
# The pass bar, from the design, stated before any number is collected.
SECOND_AXIS_GAIN = 1.15
RHO_MAX = 0.6
YAW_MISS_MAX_PCT = 20.0
MIN_GRABS = sc.MIN_SAMPLES
# How far before the fist closes a clean face row may be and still count.
YAW_WINDOW_S = 1.0


def _pct(values, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (1.0 - pos + low) + ordered[high] * (pos - low)


def _spread(values) -> float:
    """A ROBUST spread: the IQR over 1.349. Medians and IQRs throughout,
    never means and standard deviations, because the label is a proxy and a
    minority of wrong ones must move nothing."""
    if len(values) < 4:
        return 0.0
    return (_pct(values, 0.75) - _pct(values, 0.25)) / sc.IQR_TO_SIGMA


def _pearson(xs, ys) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx = statistics.fmean(xs[:n])
    my = statistics.fmean(ys[:n])
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs[:n], ys[:n]))
    sxx = sum((x - mx) ** 2 for x in xs[:n])
    syy = sum((y - my) ** 2 for y in ys[:n])
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _mahalanobis(a, b) -> float:
    """The 2-D separation using the MEASURED 2x2 covariance.

    Never assumed diagonal. I ran the combination formula and it does
    something worth flagging: for d_hand 3.8 and d_yaw 6.8, rho = 0 gives
    7.8, rho = 0.5 gives 6.8 (yaw alone, the hand contributing nothing) and
    rho = 0.9 gives 8.6 -- correlated noise can HELP when the class means
    do not lie along the correlation direction. Which of those he is in is
    a fact about his body, so it is measured and used as measured.
    """
    ax, ay = a["hand"], a["yaw"]
    bx, by = b["hand"], b["yaw"]
    if len(ax) < 3 or len(bx) < 3:
        return 0.0
    pooled = []
    for xs, ys in ((ax, ay), (bx, by)):
        mx, my = statistics.median(xs), statistics.median(ys)
        pooled.append([(x - mx, y - my) for x, y in zip(xs, ys)])
    rows = pooled[0] + pooled[1]
    n = len(rows)
    sxx = sum(dx * dx for dx, _ in rows) / max(n - 1, 1)
    syy = sum(dy * dy for _, dy in rows) / max(n - 1, 1)
    sxy = sum(dx * dy for dx, dy in rows) / max(n - 1, 1)
    det = sxx * syy - sxy * sxy
    if det <= 1e-12:
        return 0.0
    dx = statistics.median(ax) - statistics.median(bx)
    dy = statistics.median(ay) - statistics.median(by)
    inv = ((syy / det, -sxy / det), (-sxy / det, sxx / det))
    q = (dx * (inv[0][0] * dx + inv[0][1] * dy)
         + dy * (inv[1][0] * dx + inv[1][1] * dy))
    return math.sqrt(max(q, 0.0))


# ------------------------------------------------------------- the maths
def summarise(rows: dict, misses: int, looks: int) -> dict:
    """``{label: [(hand_x_u, yaw_t or None), ...]}`` -> the seven numbers.

    ``rows`` is what a run collected; nothing about a frame is in it.
    """
    per = {}
    for label, samples in rows.items():
        hand = [h for h, _y in samples]
        yaw = [y for _h, y in samples if y is not None]
        per[label] = {
            "n": len(hand), "n_yaw": len(yaw),
            "hand": hand, "yaw": yaw,
            "hand_p50": round(statistics.median(hand), 4) if hand else 0.0,
            "hand_sigma": round(_spread(hand), 4),
            "yaw_p10": round(_pct(yaw, 0.10), 4),
            "yaw_p50": round(statistics.median(yaw), 4) if yaw else 0.0,
            "yaw_p90": round(_pct(yaw, 0.90), 4),
            "yaw_sigma": round(_spread(yaw), 4),
            "rho": round(_pearson([h for h, y in samples if y is not None],
                                  yaw), 3),
        }
    sigma_u = statistics.fmean(
        [p["hand_sigma"] for p in per.values() if p["hand_sigma"] > 0]) \
        if any(p["hand_sigma"] > 0 for p in per.values()) else 0.0
    sigma_t = statistics.fmean(
        [p["yaw_sigma"] for p in per.values() if p["yaw_sigma"] > 0]) \
        if any(p["yaw_sigma"] > 0 for p in per.values()) else 0.0
    pairs = {}
    names = sorted(per)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d_hand = (abs(per[a]["hand_p50"] - per[b]["hand_p50"]) / sigma_u
                      if sigma_u > 0 else 0.0)
            d_yaw = (abs(per[a]["yaw_p50"] - per[b]["yaw_p50"]) / sigma_t
                     if sigma_t > 0 else 0.0)
            pairs["%s|%s" % (a, b)] = {
                "d_hand": round(d_hand, 3), "d_yaw": round(d_yaw, 3),
                "d_2D": round(_mahalanobis(per[a], per[b]), 3),
                "same_machine": MACHINE_OF.get(a) == MACHINE_OF.get(b),
            }
    for p in per.values():                    # the raw lists do not survive
        p.pop("hand", None)
        p.pop("yaw", None)
    miss_pct = round(100.0 * misses / looks, 1) if looks else 0.0
    return {"per_label": per, "pairs": pairs,
            "sigma_hand_u": round(sigma_u, 4), "sigma_yaw_t": round(sigma_t, 4),
            "grabs": looks, "yaw_misses": misses, "yaw_miss_pct": miss_pct}


def zones(sigma_u: float) -> dict:
    """How often a 2-zone and a 3-zone split would be right, at a MEASURED
    placement sigma. The span is his own lens: 4.57 hand-units at a 450 mm
    reach, which agrees with the 4.6 the first design pass used to 1%."""
    span = 4.57
    out = {"span_u": span, "sigma_u": round(float(sigma_u), 4)}
    for k in (2, 3):
        half = span / k / 2.0
        if sigma_u <= 0.0:
            out["zones_%d_pct" % k] = 0.0
            continue
        z = half / sigma_u
        # P(|N(0,1)| < z), the middle zone -- the hardest one.
        p = math.erf(z / math.sqrt(2.0))
        out["zones_%d_half_u" % k] = round(half, 4)
        out["zones_%d_pct" % k] = round(100.0 * p, 3)
    return out


def verdict(summary: dict) -> list:
    """(name, ok, detail, measurable). The bar, applied."""
    rows = []
    per = summary["per_label"]
    thin = [k for k, v in per.items() if v["n"] < MIN_GRABS]
    enough = per and not thin
    rows.append(("samples", enough,
                 "%d label(s), %s" % (len(per),
                                      "thin: " + ", ".join(sorted(thin))
                                      if thin else "%d+ each" % MIN_GRABS),
                 True))
    miss = summary["yaw_miss_pct"]
    rows.append(("yaw available", miss < YAW_MISS_MAX_PCT,
                 "yaw_miss_pct %.1f%% (bar %.0f%%)" % (miss, YAW_MISS_MAX_PCT),
                 summary["grabs"] > 0))
    cross = [(k, v) for k, v in summary["pairs"].items()
             if not v["same_machine"]]
    if cross:
        worst = min(cross, key=lambda kv: max(kv[1]["d_hand"], kv[1]["d_yaw"]))
        name, p = worst
        rows.append(("machines apart",
                     max(p["d_hand"], p["d_yaw"]) >= 2.0 * sc.MIN_PAIR_SIGMA,
                     "%s: hand %.2f sigma, yaw %.2f sigma"
                     % (name, p["d_hand"], p["d_yaw"]), enough))
        gain = (p["d_2D"] / max(p["d_hand"], p["d_yaw"], 1e-9))
        rows.append(("yaw is a 2nd axis", gain >= SECOND_AXIS_GAIN,
                     "d_2D %.2f is %.2fx the better single axis (bar %.2f)"
                     % (p["d_2D"], gain, SECOND_AXIS_GAIN), enough))
    rhos = [abs(v["rho"]) for v in per.values() if v["n_yaw"] >= 3]
    if rhos:
        rows.append(("axes independent", max(rhos) < RHO_MAX,
                     "|rho| max %.2f (bar %.2f)" % (max(rhos), RHO_MAX),
                     enough))
    return rows


# ------------------------------------------------------------ the run
def synthetic_rows(n: int = 60, sigma_u: float = 0.30,
                   sigma_t: float = 0.05, seed: int = 7) -> tuple:
    """Rows this process invents, so the maths above can be exercised with
    no camera and no him. A drawn number is not a measurement, and nothing
    this produces says anything about his hand or his neck."""
    import random
    rng = random.Random(seed)
    centres = {"right": (1.34, -0.30), "middle": (-0.63, 0.02),
               "left": (-1.90, 0.34)}
    rows = {}
    misses = 0
    looks = 0
    for label, (hu, yt) in centres.items():
        got = []
        for i in range(n):
            looks += 1
            yaw = None
            if i % 9:                            # ~11% with no clean face
                yaw = rng.gauss(yt, sigma_t)
            else:
                misses += 1
            got.append((rng.gauss(hu, sigma_u), yaw))
        rows[label] = got
    return rows, misses, looks


def live_rows(cfg, policy, *, label: str, minutes: float, say) -> tuple:
    """The real loop. One row per GRAB: the lateral hand position in
    hand-units and the newest clean yaw sample from the second BEFORE the
    fist closed -- never at the grab instant, because by then the reaching
    arm is across his face."""
    detector, det_why = cam.detector_from_config(
        cfg, score_threshold=facedetect.PROBE_THRESHOLD)
    if detector is None:
        say("   NO FACE DETECTOR (%s): the reach ratio is 0.0 and nothing "
            "can grab." % det_why)
        return {}, 0, 0
    tracker = hp.HandTracker(
        model_dir=str(cfg.get("gesture.model_dir", "") or "") or None,
        threads=int(cfg.get("gesture.hand_threads",
                            cfg.get("camera.threads", 2))))
    lens = cam.lens_from_config(cfg)
    head = cam.head_from_config(cfg)
    mirrored = bool(cfg.get("camera.mirrored", False))
    fps = float(cfg.get("camera.preview_fps", 6.0))
    thresholds = g.CastThresholds.for_fps(fps)
    events: list = []
    machine = g.CastGesture(thresholds, (lens.width_px, lens.height_px),
                            mirrored=mirrored, on_event=events.append,
                            preview_fps=fps)
    feed, why = cam.build(cfg, policy)
    if feed is None:
        say("   NO FEED: %s" % why)
        return {}, 0, 0
    dw, dh = getattr(detector, "input_size", (0, 0))
    sx = lens.width_px / float(dw) if dw else 1.0
    sy = lens.height_px / float(dh) if dh else 1.0
    eyes: collections.deque = collections.deque()
    yaws: collections.deque = collections.deque(maxlen=32)
    rows = {label: []}
    misses = grabs = 0
    start = time.monotonic()
    seq = 0
    feed.start()
    try:
        while time.monotonic() - start < minutes * 60.0:
            frame = feed.frame()
            if frame is None:
                time.sleep(1.0 / max(fps, 1.0))
                continue
            now = time.monotonic()
            seq += 1
            obs = [vr.observe(r, lens, sx, sy, head)
                   for r in (detector.detect(frame) or [])]
            eye_px = 0.0
            if len(obs) == 1 and obs[0].landmarks_ok and obs[0].eye_px > 0:
                eyes.append((now, float(obs[0].eye_px)))
                yaws.append((now, float(obs[0].yaw_t)))
            while eyes and now - eyes[0][0] > 5.0:
                eyes.popleft()
            if eyes:
                eye_px = statistics.median(e for _t, e in eyes)
            hands = tuple(g.observe_hand(r.lm, eye_px, r.conf)
                          for r in tracker.detect(frame))
            ev = machine.update(hands, eye_px, seq)
            del frame                        # the last reference, explicitly
            if ev is not None and ev.kind == "grab" and hands:
                grabs += 1
                yaw = None
                for t, value in reversed(yaws):
                    if t <= now and (now - t) <= YAW_WINDOW_S:
                        yaw = value
                        break
                if yaw is None:
                    misses += 1
                rows[label].append((
                    sc.hand_x_u(hands[0].cx, hands[0].palm_diag,
                                lens.width_px, mirrored), yaw))
                say("   grab %3d  hand_x_u %+.3f  yaw_t %s"
                    % (grabs, rows[label][-1][0],
                       "%+.3f" % yaw if yaw is not None else "   -   "))
    finally:
        feed.stop()
    return rows, misses, grabs


def report_rows(summary: dict, say) -> None:
    say("")
    say("   per label")
    say("   %-8s %5s %5s | %8s %8s | %8s %8s %8s %8s | %6s"
        % ("label", "n", "n_yaw", "hand_p50", "hand_sig",
           "yaw_p10", "yaw_p50", "yaw_p90", "yaw_sig", "rho"))
    for label in sorted(summary["per_label"]):
        p = summary["per_label"][label]
        say("   %-8s %5d %5d | %+8.3f %8.3f | %+8.3f %+8.3f %+8.3f %8.3f "
            "| %+6.2f"
            % (label, p["n"], p["n_yaw"], p["hand_p50"], p["hand_sigma"],
               p["yaw_p10"], p["yaw_p50"], p["yaw_p90"], p["yaw_sigma"],
               p["rho"]))
    say("")
    say("   pairs (separation in sigmas; * crosses the MACHINE boundary, "
        "which is the only pair that has to work)")
    for name in sorted(summary["pairs"]):
        p = summary["pairs"][name]
        say("   %-16s hand %6.2f   yaw %6.2f   2-D %6.2f   %s"
            % (name.replace("|", " vs "), p["d_hand"], p["d_yaw"],
               p["d_2D"], "" if p["same_machine"] else "*"))
    z = zones(summary["sigma_hand_u"])
    say("")
    say("   zones at the measured placement sigma %.3f u (span %.2f u)"
        % (z["sigma_u"], z["span_u"]))
    for k in (2, 3):
        say("   %d zones: half-width %.3f u -> %.2f%% correct in the "
            "hardest zone" % (k, z.get("zones_%d_half_u" % k, 0.0),
                              z.get("zones_%d_pct" % k, 0.0)))
    say("   yaw_miss_pct %.1f%% over %d grabs -- LOOK AT THIS FIRST"
        % (summary["yaw_miss_pct"], summary["grabs"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=10.0,
                    help="how long to watch (default 10)")
    ap.add_argument("--label", default="right", choices=list(LABELS),
                    help="which screen he is working at for this run")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the arithmetic over invented rows. Opens no "
                         "device and says nothing about him.")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable, and just as pixel-free")
    args = ap.parse_args(argv)

    report: dict = {"banner": BANNER}
    say = (lambda *a: None) if args.json else print
    say(BANNER)
    say("")

    if args.synthetic:
        rows, misses, looks = synthetic_rows()
        say("1. SYNTHETIC rows (no device, and nothing about him)")
        summary = summarise(rows, misses, looks)
        report["summary"] = summary
        report_rows(summary, say)
        return _finish(report, args, summary, say)

    cfg = AssistantConfig.load()
    policy = SensingPolicy(cfg=cfg)
    status = policy.status()
    report["sensing"] = status
    say("1. sensing")
    say("   camera=%s offline=%s reason=%s curfew=%s"
        % (status["camera"], status["offline"], status["reason"] or "-",
           status.get("curfew", "") or "-"))
    if not status["camera"]:
        say("")
        say("STOPPED: sensing says the camera may not run (%s). The device "
            "was not opened." % (status["reason"] or "denied"))
        report["summary"] = {}
        return _finish(report, args, {}, say, code=2)
    if not cam.device_nodes():
        say("")
        say("STOPPED: no /dev/video* on this box.")
        report["summary"] = {}
        return _finish(report, args, {}, say, code=3)

    say("")
    say("2. watching you work at the %s screen for %.1f minutes"
        % (args.label, args.minutes))
    say("   Reach out and grab as you normally would. Nothing is shown, "
        "nothing is saved.")
    rows, misses, looks = live_rows(cfg, policy, label=args.label,
                                    minutes=args.minutes, say=say)
    summary = summarise(rows, misses, looks)
    report["summary"] = summary
    report_rows(summary, say)
    return _finish(report, args, summary, say)


def _finish(report: dict, args, summary: dict, say, code: int = 0) -> int:
    rows = verdict(summary) if summary else []
    report["checks"] = [{"name": n, "ok": bool(ok), "detail": d,
                         "measurable": bool(m)} for n, ok, d, m in rows]
    if rows:
        say("")
        say("   checks")
        for name, ok, detail, measurable in rows:
            mark = "PASS " if ok else "FAIL "
            say("   %s %-18s %s" % (mark if measurable else "n/a  ",
                                    name, detail))
            if measurable and not ok:
                code = code or 1
    report["exit_code"] = code
    vr.assert_numbers_only(report)              # the promise, enforced
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
