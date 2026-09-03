#!/usr/bin/env python3
"""One command that answers "will the camera work here", in numbers only.

    ~/vss_env/bin/python scripts/vision_selfcheck.py
    ~/vss_env/bin/python scripts/vision_selfcheck.py --frames 120 --json

RUN THIS WHEN THE ADAPTER ARRIVES. It reports, in this order:

 1. what the sensing owner says -- offline mode, the 21:00-07:00 curfew, the
    fail-safe. If the camera is not allowed, NOTHING below runs and the
    device is not opened. That is not a limitation of the script, it is the
    same rule the running Jarvis obeys, reached through the same object;
 2. which face models are on disk, their sizes against the pinned ones,
    their licences, and whether this OpenCV can even build them;
 3. which /dev/video* nodes exist, and what formats and frame rates they
    advertise (via v4l2-ctl when it is installed);
 4. the geometry the configured field of view implies -- how wide the
    frame is at 95 cm, how many pixels a 16 cm face gets, and how many of
    those survive the downscale to the detector;
 5. the mode the driver actually GRANTED against the one the config asked
    for -- v4l2 answers an impossible request by quietly giving you
    something else, which is how "width: 1920" sat in the config for a 720p
    camera without anybody finding out;
 6. whether autofocus can be switched off and focus pinned;
 7. what the detector sees on HIS scene: detection counts, confidences, box
    sizes in both capture and detector pixels, bearings and head angles
    derived from the CONFIGURED field of view, frame level and contrast,
    per-stage timings, and a pass/fail line against each threshold in his
    config.

NO IMAGE DATA LEAVES THIS SCRIPT. It prints no pixels, saves no frame, opens
no window, and every number it prints is a count, an angle, a size, a score
or a millisecond. That is deliberate and it is the point: Hunter can run this
against his real camera and paste the output to anyone, including me, without
either of us ever seeing his room. ``jarvis/visionrig.assert_numbers_only``
is run over the report before it is printed, so this is enforced rather than
promised.

Exit codes: 0 everything checked out, 1 something failed a check, 2 sensing
said no (offline or curfew), 3 there is no camera to check.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis import camera as cam                          # noqa: E402
from jarvis import facedetect                             # noqa: E402
from jarvis import visionrig as vr                        # noqa: E402
from jarvis.assistant_config import AssistantConfig       # noqa: E402
from jarvis.sensing import SensingPolicy                   # noqa: E402

BANNER = (
    "vision self-check -- numbers only. No frame is displayed, saved or "
    "described;\nevery line below is a count, an angle, a size, a score or "
    "a millisecond.")


def cam_describe(lens, detect_width: int) -> str:
    from jarvis.facemodels import describe          # noqa: PLC0415
    return describe(lens, 95.0, detect_width)


def v4l2_formats(node: str, timeout: float = 5.0) -> dict:
    """What the driver advertises. Optional: v4l2-ctl is not installed here.

    Parsed loosely on purpose -- the value is the resolution and frame-rate
    list, and a strict parser that broke on a new v4l2-utils would cost the
    whole section for nothing.
    """
    try:
        out = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats-ext"],
                             capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc), "modes": []}
    if out.returncode != 0:
        return {"available": False, "reason": out.stderr.strip()[:200],
                "modes": []}
    modes, fourcc = [], ""
    for line in out.stdout.splitlines():
        text = line.strip()
        if text.startswith("[") and "'" in text:
            fourcc = text.split("'")[1]
        elif text.startswith("Size:"):
            parts = text.split()
            modes.append({"fourcc": fourcc,
                          "size": parts[-1] if parts else "", "fps": []})
        elif text.startswith("Interval:") and modes:
            # "Interval: Discrete 0.033s (30.000 fps)" -- the rate is the
            # token BEFORE "fps", and older v4l2-utils glue them together.
            tokens = text.replace("(", " ").replace(")", " ").split()
            for i, token in enumerate(tokens):
                rate = ""
                if token == "fps" and i:
                    rate = tokens[i - 1]
                elif token.endswith("fps") and len(token) > 3:
                    rate = token[:-3]
                if not rate:
                    continue
                try:
                    modes[-1]["fps"].append(float(rate))
                except ValueError:
                    pass
    return {"available": True, "reason": "", "modes": modes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=60,
                    help="frames to run through the detector (default 60)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this long instead of after --frames")
    ap.add_argument("--device", default=None,
                    help="override camera.device, e.g. /dev/video2")
    ap.add_argument("--deep", action="store_true",
                    help="hash the model files as well as sizing them")
    ap.add_argument("--models-only", action="store_true",
                    help="check the weights and stop; opens no device")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the detector on frames this process generates "
                         "instead of on a camera. Opens no device, so it "
                         "works today. Measures COST, never accuracy: a "
                         "synthetic face is not a face")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable, and just as pixel-free")
    args = ap.parse_args(argv)

    report: dict = {"banner": BANNER}
    say = (lambda *a: None) if args.json else print
    say(BANNER)
    say("")

    cfg = AssistantConfig.load()
    if args.device is not None:
        cfg.data.setdefault("camera", {})["device"] = args.device

    # ------------------------------------------------------------ 1. sensing
    policy = SensingPolicy(cfg=cfg)
    status = policy.status()
    report["sensing"] = status
    say("1. sensing")
    say("   camera=%s radar=%s offline=%s reason=%s curfew=%s failsafe=%s"
        % (status["camera"], status["radar"], status["offline"],
           status["reason"] or "-", status["curfew"] or "-",
           status["failsafe"]))

    # ------------------------------------------------------------- 2. models
    model_dir = str(cfg.get("camera.model_dir", "") or "") or None
    probe = facedetect.probe(deep=args.deep, model_dir=model_dir)
    report["models"] = probe
    say("")
    say("2. face models  (%s)" % probe["dir"])
    for key in sorted(probe["models"]):
        m = probe["models"][key]
        say("   %-10s %-5s %10d B (want %10d)  %-10s %s"
            % (key, "ok" if m["ok"] else "MISS", m["bytes"],
               m["expected_bytes"], m["licence"], m["reason"]))
    cv2s = probe["cv2"]
    say("   cv2 %s  detector_api=%s recogniser_api=%s %s"
        % (cv2s["version"] or "-", cv2s["detector_api"],
           cv2s["recogniser_api"], cv2s["reason"]))
    if args.models_only:
        return _finish(report, args, 0 if probe["ready"] else 1)

    # ------------------------------------------------------------ 3. devices
    nodes = cam.device_nodes()
    report["devices"] = {"nodes": nodes, "formats": {}}
    say("")
    say("3. devices")
    if not nodes:
        say("   no /dev/video* on this box")
    for node in nodes:
        fmt = v4l2_formats(node)
        report["devices"]["formats"][node] = fmt
        if not fmt["available"]:
            say("   %s  (v4l2-ctl unavailable: %s)" % (node, fmt["reason"]))
            continue
        say("   %s" % node)
        for mode in fmt["modes"]:
            rates = ", ".join("%.0f" % f for f in mode["fps"]) or "-"
            say("      %-6s %-12s fps %s"
                % (mode["fourcc"], mode["size"], rates))

    # ----------------------------------------------------------- geometry
    try:
        lens_cfg = cam.lens_from_config(cfg)
    except ValueError as exc:
        report["geometry_error"] = str(exc)
        say("")
        say("STOPPED: %s" % exc)
        return _finish(report, args, 1)
    detect_w = int(cfg.get("camera.detect_width", 320))
    geom = {"width": lens_cfg.width_px, "height": lens_cfg.height_px,
            "hfov_deg": lens_cfg.hfov_deg,
            "span_cm_at_95": lens_cfg.span_cm(95.0),
            "px_per_cm_at_95": lens_cfg.px_per_cm(95.0),
            "face_px_at_95": lens_cfg.face_px(95.0),
            "detect_face_px_at_95": lens_cfg.detect_face_px(95.0, detect_w),
            "detect_width": detect_w,
            "detect_height": int(cfg.get("camera.detect_height", 180))}
    report["geometry"] = geom
    say("")
    say("4. geometry from the CONFIGURED lens, at a 95 cm mount")
    say("   %s" % cam_describe(lens_cfg, detect_w))
    advertised = [m["size"] for f in report.get("devices", {})
                  .get("formats", {}).values() for m in f.get("modes", [])]
    want = "%dx%d" % (lens_cfg.width_px, lens_cfg.height_px)
    if advertised and want not in advertised:
        say("   WARNING: %s is not a mode any node advertised (%s). v4l2 "
            "answers an impossible request by quietly granting something "
            "else, so section 5 is the one to believe."
            % (want, ", ".join(sorted(set(advertised)))))
    if geom["detect_face_px_at_95"] < 10.0:
        say("   WARNING: a 16 cm face is %.0f px at the detector, under "
            "YuNet's ~10 px floor. This mount cannot work."
            % geom["detect_face_px_at_95"])
    if geom["face_px_at_95"] < 112.0:
        say("   WARNING: a 16 cm face is %.0f px at capture, under SFace's "
            "112x112 input. Identity will be unreliable."
            % geom["face_px_at_95"])


    if args.synthetic:
        say("")
        say("7. what the detector costs on SYNTHETIC frames  (%d frames)"
            % args.frames)
        say("   No device is opened. These are TIMINGS and SIZES; the "
            "confidence lines below describe generated noise, not a face, "
            "and say nothing about whether yours would be detected.")
        detector, det_why = cam.detector_from_config(
            cfg, score_threshold=facedetect.PROBE_THRESHOLD)
        source = vr.SyntheticSource(lens_cfg.width_px, lens_cfg.height_px,
                                    frames=args.frames)
        result = vr.Rig(source, detector, lens_cfg,
                        cam.thresholds_from_config(cfg),
                        head=cam.head_from_config(cfg),
                        detector_reason=det_why).run(frames=args.frames,
                                                     seconds=args.seconds)
        payload = result.to_dict()
        vr.assert_numbers_only(payload)
        report["rig"] = payload
        for line in result.lines():
            say("   " + line)
        return _finish(report, args, 0 if result.detector_ok else 1)

    if not status["camera"]:
        say("")
        say("STOPPED: sensing says the camera may not run (%s). The device "
            "was not opened." % (status["reason"] or "denied"))
        say('Say "come back online" to Jarvis, or wait for the curfew to '
            "end, and run this again.")
        return _finish(report, args, 2)
    if not nodes:
        return _finish(report, args, 3)

    # ------------------------------------------------- 5/6. mode and focus
    feed, why = cam.build(cfg, policy)
    if feed is None:
        # camera.enabled is False on a fresh config, and that must not stop a
        # bring-up check -- it is the thing he runs BEFORE turning it on.
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
    asked = {"width": lens.width_px, "height": lens.height_px,
             "hfov_deg": lens.hfov_deg,
             "fourcc": str(cfg.get("camera.fourcc", "MJPG") or "")}
    report["asked"] = asked
    say("")
    say("5. the mode the driver granted")
    dev = feed.gate.open()
    if dev is None:
        say("   the gate declined to open the device")
        return _finish(report, args, 2)
    try:
        granted = cam.capture_mode(dev)
        focus = cam.focus_probe(dev)
    finally:
        feed.gate.release()
    report["granted"] = granted
    report["focus"] = focus
    say("   asked   %dx%d %s" % (asked["width"], asked["height"],
                                 asked["fourcc"]))
    say("   granted %.0fx%.0f %s at %.1f fps"
        % (granted["width"], granted["height"], granted["fourcc"] or "?",
           granted["fps"]))
    mode_ok = (granted["width"] == float(asked["width"])
               and granted["height"] == float(asked["height"]))
    if not mode_ok:
        say("   MISMATCH: the driver silently granted a different mode. Set "
            "camera.width/height to what it granted, and re-derive "
            "camera.hfov_deg for THAT mode -- cropping changes the field of "
            "view, scaling does not.")
    say("")
    say("6. focus")
    for name in ("autofocus", "focus"):
        f = focus[name]
        say("   %-10s before %.1f  set_accepted=%s  after %.1f  pinned=%s"
            % (name, f["before"], f["set_accepted"], f["after"], f["pinned"]))

    # ------------------------------------------------------------ 6. the rig
    say("")
    say("7. what the detector sees  (%d frames)" % args.frames)
    detector, det_why = cam.detector_from_config(
        cfg, score_threshold=facedetect.PROBE_THRESHOLD)
    recogniser = None
    if bool(cfg.get("camera.identity", False)):
        recogniser, _rwhy = cam.recogniser_from_config(cfg)
    rig = vr.Rig(cam.FeedSource(feed), detector, lens,
                 cam.thresholds_from_config(cfg),
                 recogniser=recogniser, head=cam.head_from_config(cfg),
                 detector_reason=det_why)
    result = rig.run(frames=args.frames, seconds=args.seconds)
    feed.close()
    payload = result.to_dict()
    vr.assert_numbers_only(payload)          # the promise, enforced
    report["rig"] = payload
    for line in result.lines():
        say("   " + line)

    say("")
    say("The detector floored at %.2f while your camera.min_conf is %.2f, on "
        "purpose: a face scoring under your bar is reported WITH ITS SCORE "
        "rather than not reported at all." % (facedetect.PROBE_THRESHOLD,
                                              float(cfg.get(
                                                  "camera.min_conf", 0.6))))
    code = 0 if (result.ok and mode_ok) else 1
    return _finish(report, args, code)


def _finish(report: dict, args, code: int) -> int:
    report["exit_code"] = code
    if args.json:
        vr.assert_numbers_only(report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
