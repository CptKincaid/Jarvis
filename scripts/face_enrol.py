#!/usr/bin/env python3
"""Enrol your face, in numbers only -- and back it up, check it, or destroy it.

    ~/vss_env/bin/python scripts/face_enrol.py                 # enrol
    ~/vss_env/bin/python scripts/face_enrol.py --status        # no camera
    ~/vss_env/bin/python scripts/face_enrol.py --verify        # does it match?
    ~/vss_env/bin/python scripts/face_enrol.py --backup ~/face-backup
    ~/vss_env/bin/python scripts/face_enrol.py --restore ~/face-backup
    ~/vss_env/bin/python scripts/face_enrol.py --rollback      # undo the last
    ~/vss_env/bin/python scripts/face_enrol.py --delete        # destroy it
    ~/vss_env/bin/python scripts/face_enrol.py --append        # add to it
    ~/vss_env/bin/python scripts/face_enrol.py --reset         # replace it

NO IMAGE DATA LEAVES THIS SCRIPT. It prints no pixels, saves no frame, opens
no window, writes no crop and shows nothing. Every line it prints is a count,
an angle, a size, a score, a millisecond or a verdict, and
``jarvis/visionrig.assert_numbers_only`` is run over the whole report before
anything is printed, so that is enforced rather than promised. **That is the
entire workflow**: you run this, you paste the output, and somebody who has
never seen you can tell you whether the enrolment is good.

The only thing that reaches the disk is the 128-float embedding, into
``~/.aiws_trainer/face_gallery/`` at 0600 in a 0700 directory.

WHAT IT ASKS YOU TO DO, AND WHY IT IS NOT ONE POSE. Measured on your camera
2026-09-02, your head is at ~14 deg of yaw when you look at the lens and ~54
deg when you look at your screen. A gallery built at one head position stops
working the moment you turn to work -- and it fails SILENTLY, which is the
expensive way for this to be wrong. So the run walks five stations across the
yaw range you actually occupy, and it REFUSES to save a pool that did not
achieve the spread. The precedent is your voiceprint: 11 ECAPA embeddings
with pairwise cosine 0.289-0.735, a real spread rather than eleven readings
of one sentence.

WHAT IT REFUSES TO SAVE, and it says which:

* **too tight** -- near-duplicate samples, or no pose variation. This gallery
  would know you in one position and reject you in every other;
* **too loose** -- a sample that does not cluster with the rest, which is
  either somebody else who walked through frame or a crop that is not a face.
  SFace scores non-faces CONFIDENTLY (jarvis/facemodels.py), so a poisoned
  sample does not look weak, it looks certain -- and ``FaceGallery.match``
  scores against the pool's BEST member, so one is enough.

SENSING OWNS THE LENS, NOT THIS SCRIPT. Offline mode, the 21:00-07:00 curfew
and the fail-safe are read from ``jarvis/sensing.py`` and the camera is opened
through ``jarvis/camera.CameraFeed``, so this obeys exactly the rules the
running Jarvis obeys, through the same objects. If sensing says no, nothing
below it runs and no device is opened. If the curfew arrives mid-enrolment,
the frames stop and the run says so rather than finishing quietly.

BACKUP, SAID PLAINLY. The gallery is generational -- a save never overwrites,
it writes gen-00002.npz beside gen-00001.npz -- and that makes a BAD WRITE
recoverable. **It is not a backup.** Every generation lives in the same
directory on the same disk, so one disk failure takes all of them at once,
and there is no backup system on this box: restic is installed, no repository
exists, and nothing under ~/.aiws_trainer is backed up. ``--backup`` is the
only thing that puts your enrolment somewhere else, and it is only as good as
where you point it -- another directory on the same disk is a copy, not a
backup.

ENROLLING AGAIN, THREE WAYS, AND WHAT EACH ONE DESTROYS.

* plain re-run -- captures a fresh pool and saves it as a NEW generation.
  Nothing is destroyed; the previous enrolment is one --rollback away. This
  is the one you want.
* ``--append`` -- loads the existing pool first and saves loaded+new. The
  checks judge the MERGED pool, because that is what reaches the disk: a
  batch that is somebody else fails cohesion over the 26 embeddings that
  would have landed, not over the 13 it captured.
* ``--reset`` -- captures first, saves, and only then destroys the
  generations that predate the new one. It asks you to type "reset", it says
  how many are going, and if the run fails a check it destroys NOTHING.

Exit codes: 0 ok, 1 something failed a check, 2 sensing said no, 3 there is
nothing to work with (no camera, no models, no gallery).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis import camera as cam                          # noqa: E402
from jarvis import facedetect                             # noqa: E402
from jarvis import faceenrol as fe                        # noqa: E402
from jarvis import visionrig as vr                        # noqa: E402
from jarvis.assistant_config import AssistantConfig       # noqa: E402
from jarvis.config import PATHS                           # noqa: E402
from jarvis.eye import FaceIdentifier                     # noqa: E402
from jarvis.facegallery import (SFACE_COSINE_SAME,        # noqa: E402
                                FaceGallery)
from jarvis.sensing import SensingPolicy                   # noqa: E402

BANNER = (
    "face enrolment -- numbers only. No frame is displayed, saved, described\n"
    "or written; every line below is a count, an angle, a size, a score or a\n"
    "millisecond, and the only thing that reaches the disk is a 128-float\n"
    "embedding.")

BACKUP_WARNING = (
    "NOTE ON BACKUP: the generations below all live in the SAME directory on\n"
    "the SAME disk. They make a bad write recoverable; they do not survive a\n"
    "disk failure, and nothing under ~/.aiws_trainer is backed up on this\n"
    "box (restic is installed, no repository exists). Run --backup to a\n"
    "different disk if this enrolment is worth keeping.")


# --------------------------------------------------------------- the seams
def open_gallery() -> FaceGallery:
    """The user's gallery. ``PATHS.FACE_GALLERY`` honours
    ``JARVIS_FACE_GALLERY``, which tests/conftest.py forces into a throwaway
    directory -- so nothing in the suite can reach the real one."""
    return FaceGallery(root=PATHS.FACE_GALLERY)


def build_models(cfg):
    """``(detector, recogniser, reason)`` -- never raises, never falls back.

    The detector is floored LOW rather than at his ``camera.min_conf``, for
    the same reason scripts/vision_selfcheck.py does it: a face scoring 0.45
    against a 0.6 bar must be reported WITH ITS SCORE, not vanish and read as
    "no face seen". The bar is then applied by the quality gate, which says
    which bar it was.
    """
    detector, why = cam.detector_from_config(
        cfg, score_threshold=facedetect.PROBE_THRESHOLD)
    if detector is None:
        return None, None, why
    try:
        rec = facedetect.load_recogniser(
            min_conf=float(cfg.get("camera.min_conf", 0.6)),
            model_dir=str(cfg.get("camera.model_dir", "") or "") or None)
    except Exception as exc:  # noqa: BLE001 - absence is not a crash
        return detector, None, str(exc)
    return detector, rec, ""


def build_feed(cfg, policy):
    """``(feed, reason)``. Falls back to constructing the feed by hand when
    ``camera.enabled`` is false, because enrolment is a thing he does BEFORE
    turning the camera on -- the same accommodation the self-check makes."""
    feed, why = cam.build(cfg, policy)
    if feed is not None:
        return feed, ""
    try:
        lens = cam.lens_from_config(cfg)
        device = str(cfg.get("camera.device", "") or "")
        fourcc = str(cfg.get("camera.fourcc", cam.DEFAULT_FOURCC) or "")
        return cam.CameraFeed(
            policy,
            lambda: cam.open_capture(device, lens.width_px, lens.height_px,
                                     fourcc),
            lens=lens, present=lambda: cam.device_present(device)), why
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def owner_label(cfg) -> str:
    """The one label this script will ever write.

    HIS RULING, not a limitation: enrol HIM only, no family. A second face in
    the gallery is a second person the wake gate can be asked about, and the
    governing rule is that identity may REMOVE capability or ADD a name and
    must never GRANT capability the existing gates do not already grant -- a
    rule that gets harder to hold the more names exist.
    """
    name = str(cfg.get("user.name", "") or "hunter").strip().lower()
    keep = "".join(c if (c.isalnum() or c in "-_") else "" for c in name)
    return keep or "hunter"


# -------------------------------------------------------------- the modes
def do_status(cfg, gallery: FaceGallery, say) -> tuple:
    """Everything about the stored enrolment, with no camera opened."""
    gens = gallery.generations()
    say("gallery    %s" % gallery.root)
    say("           %d generation(s): %s"
        % (len(gens), ", ".join(str(g) for g in gens) or "-"))
    # harden() REPAIRS as well as reports -- his voiceprint.npz is 0664
    # today, which is how this class of mistake is found at all -- so the
    # line says which mode it found, and the repair is stated rather than
    # silent.
    modes = fe.harden(gallery.root) if gallery.root.exists() else {}
    for name in sorted(modes):
        want = 700 if name == "dir" else 600
        say("           %-16s mode %04d%s"
            % (name, modes[name],
               "  -> repaired to %04d" % want if modes[name] != want else ""))
    payload: dict = {"root": str(gallery.root), "generations": list(gens),
                     "modes": {k: int(v) for k, v in modes.items()},
                     "identity_enabled": bool(cfg.get("camera.identity",
                                                      False)),
                     "identity_min": float(cfg.get("camera.identity_min",
                                                   SFACE_COSINE_SAME))}
    if not gens:
        say("           nothing enrolled yet -- run this script with no "
            "arguments")
        payload["total"] = 0
        return 3, payload
    for gen in gens:
        one = FaceGallery(root=gallery.root)
        if not one.load(generation=gen):
            say("  gen %05d  UNREADABLE" % gen)
            continue
        prov = one.provenance()
        say("  gen %05d  %d embeddings over %d label(s)  reason=%r"
            % (gen, one.total(), len(one.labels()), prov.get("reason", "")))
    gallery.load()
    payload["loaded_generation"] = gallery.loaded_generation
    payload["total"] = gallery.total()
    payload["labels"] = list(gallery.labels())
    say("")
    say("loaded     generation %d, %d embeddings, labels %s"
        % (gallery.loaded_generation, gallery.total(),
           ", ".join(gallery.labels()) or "-"))
    for label in gallery.labels():
        embs = gallery.embeddings(label)
        pairs = fe.pairwise_cosines(embs)
        coh = fe.cohesion(embs)
        payload["pairs"] = len(pairs)
        payload["cos_min"] = min(pairs) if pairs else 0.0
        payload["cos_p50"] = vr._pct(pairs, 0.5)
        payload["cos_max"] = max(pairs) if pairs else 0.0
        payload["cohesion_min"] = min(coh) if coh else 0.0
        say("  %-10s %d embeddings, %d pairs, cosine min %.3f p50 %.3f "
            "max %.3f, worst cohesion %.3f"
            % (label, len(embs), len(pairs),
               payload["cos_min"], payload["cos_p50"], payload["cos_max"],
               payload["cohesion_min"]))
    say("")
    say("identity   camera.identity=%s  camera.identity_min=%.3f"
        % (payload["identity_enabled"], payload["identity_min"]))
    say("")
    say(BACKUP_WARNING)
    return 0, payload


def do_enrol(cfg, policy, gallery: FaceGallery, args, say) -> tuple:
    """The guided capture. Returns ``(exit_code, payload)``."""
    # SENSING IS ASKED FIRST, before the models are even looked for. It is
    # the most authoritative refusal there is, and a run that reported
    # "the weights are missing" while the real answer was "you are in the
    # curfew" would send him to fix the wrong thing.
    st = policy.status()
    if not st["camera"]:
        say("STOPPED: sensing says the camera may not run (%s). The device "
            "was not opened." % (st["reason"] or "denied"))
        say('Say "come back online" to Jarvis, or wait for the curfew to end.')
        return 2, {"reason": "sensing: %s" % (st["reason"] or "denied")}

    # THE BARS ARE BUILT BEFORE ANYTHING IS WRITTEN OR OPENED. min_conf comes
    # from a user-editable key and at 0 it removes the gate this whole lane
    # rests on, so a config that cannot produce a usable bar has to stop the
    # run BEFORE camera.identity is turned on -- a run that cannot happen must
    # not leave the switch that says his face may be written down flipped
    # behind it.
    try:
        limits = fe.SampleLimits(
            min_conf=float(cfg.get("camera.min_conf", 0.6)),
            min_face_px=float(args.min_face_px),
            min_sharpness=float(args.min_sharpness))
    except ValueError as exc:
        say("STOPPED: %s" % exc)
        say("Fix camera.min_conf in ~/.config/jarvis/assistant.json.")
        return 1, {"reason": str(exc)}

    label = owner_label(cfg)
    if not bool(cfg.get("camera.identity", False)):
        if not args.enable_identity:
            say("STOPPED: camera.identity is false, and it is the phase gate "
                "for writing anything about your face down at all.")
            say("Re-run with --enable-identity to turn it on and enrol, or "
                "set camera.identity true in ~/.config/jarvis/assistant.json.")
            return 1, {"reason": "camera.identity is false"}
        if cfg.set("camera.identity", True) is False:
            say("STOPPED: camera.identity could not be written to the config.")
            return 1, {"reason": "camera.identity could not be set"}
        say("camera.identity is now true (Jarvis must restart to read it).")

    probe = facedetect.probe(model_dir=str(cfg.get("camera.model_dir", "")
                                           or "") or None)
    if not probe["ready"]:
        for key in sorted(probe["models"]):
            m = probe["models"][key]
            if not m["ok"]:
                say("STOPPED: %s -- %s" % (key, m["reason"]))
        return 3, {"reason": "the face models are not usable"}

    detector, recogniser, why = build_models(cfg)
    if detector is None or recogniser is None:
        say("STOPPED: %s" % why)
        return 3, {"reason": why}
    feed, feed_why = build_feed(cfg, policy)
    if feed is None:
        say("STOPPED: %s" % feed_why)
        return 3, {"reason": feed_why}
    if feed_why:
        say("(%s -- enrolling anyway, this is the step that comes first)"
            % feed_why)

    # --reset DESTROYS NOTHING HERE. It used to purge before the capture, so
    # a run that then failed a check -- 'too tight' / 'too loose', the
    # outcome this whole design exists to produce -- left an empty directory
    # and no generation to roll back to, on the store the docs themselves
    # call the only copy. What it destroys and when is now: the generations
    # that predate the new one, AFTER the new one is safely on disk.
    superseded: list = []
    if args.reset:
        superseded = gallery.generations()
        if args.append:
            say("--append is ignored: --reset replaces the pool rather than "
                "adding to it.")
        if superseded and not args.yes and args.json:
            # The prompt below writes to stdout, which would land in the
            # middle of the JSON document. A destructive flag may not be
            # confirmed by a prompt nobody can see.
            say("STOPPED: --reset needs --yes in --json mode.")
            return 1, {"reason": "--reset needs --yes with --json",
                       "removed": 0}
        if superseded and not args.yes:
            say("--reset will destroy %d generation(s) of face embeddings at "
                "%s -- but only AFTER this enrolment has been captured and "
                "SAVED. If the run fails a check, or you stop it, nothing is "
                "destroyed." % (len(superseded), gallery.root))
            try:
                answer = input('Type "reset" to confirm: ').strip().lower()
            except EOFError:
                answer = ""
            if answer != "reset":
                say("Not resetting. Nothing has been captured and nothing "
                    "was destroyed.")
                return 1, {"reason": "reset not confirmed", "removed": 0}
    elif args.append and gallery.load():
        say("--append: starting from generation %d, %d embeddings. The "
            "checks below judge the MERGED pool -- the %d already stored "
            "plus whatever this run adds -- because that is what gets saved."
            % (gallery.loaded_generation, gallery.total(), gallery.total()))

    session = fe.EnrolmentSession(gallery, label, feed.lens, detector,
                                  recogniser, limits,
                                  head=cam.head_from_config(cfg))
    say("")
    say("Enrolling %r. %d stations, %d samples wanted."
        % (label, len(fe.DEFAULT_PLAN),
           sum(s.samples for s in fe.DEFAULT_PLAN)))
    say("Sit where you normally sit. Nothing you see is shown to anybody, "
        "because nothing is shown at all.")
    def wait(prompt):
        input(prompt)

    if args.auto:
        wait = None
    try:
        rep, _run = fe.run_enrolment(
            session, cam.FeedSource(feed), say=say, wait=wait,
            frames_per_station=int(args.frames_per_station),
            gap_s=float(args.gap_s),
            identity_min=float(cfg.get("camera.identity_min",
                                       SFACE_COSINE_SAME)))
    finally:
        feed.close()

    removed = 0
    if rep.ok or args.force:
        try:
            gen = gallery.save(reason="face_enrol%s" % (" --force"
                                                        if not rep.ok else ""),
                               allow_shrink=bool(args.allow_shrink))
            rep.saved_generation = gen
            rep.gallery_total = gallery.total()
            if superseded:
                removed = gallery.drop_generations(superseded)
        except ValueError as exc:
            say("")
            say("NOT SAVED: %s" % exc)
            rep.reason = rep.reason or str(exc)
    payload = rep.to_dict()
    payload["reset_removed"] = removed
    vr.assert_numbers_only(payload)
    say("")
    for line in rep.lines():
        say(line)
    if not rep.ok and not args.force:
        say("")
        say("This gallery was NOT saved. Read the FAIL line above: 'too "
            "tight' means it knows you in one pose only, 'too loose' means "
            "something in it is not you. Both are worse than no gallery, "
            "because both are silent.")
    if superseded and not rep.saved_generation:
        say("--reset destroyed NOTHING: the new enrolment was not saved, so "
            "the %d generation(s) that were there are still there."
            % len(superseded))
    elif removed:
        say("--reset: %d superseded generation(s) destroyed, now that "
            "generation %d is on disk." % (removed, rep.saved_generation))
    if rep.saved_generation:
        say("")
        say(BACKUP_WARNING)
    return (0 if (rep.ok and rep.saved_generation) else 1), payload


def do_verify(cfg, policy, gallery: FaceGallery, args, say) -> tuple:
    """Run the LIVE camera against the stored gallery and report the match
    numbers. This is the proof that the matching path works, and it is the
    only way to get that proof without looking at anything."""
    # Sensing first here too, for the same reason as in do_enrol: the most
    # authoritative refusal has to be the one he is shown.
    st = policy.status()
    if not st["camera"]:
        say("STOPPED: sensing says the camera may not run (%s). The device "
            "was not opened." % (st["reason"] or "denied"))
        return 2, {"reason": "sensing: %s" % (st["reason"] or "denied")}
    if not gallery.load():
        say("STOPPED: nothing enrolled at %s" % gallery.root)
        return 3, {"reason": "nothing enrolled"}
    detector, recogniser, why = build_models(cfg)
    if detector is None or recogniser is None:
        say("STOPPED: %s" % why)
        return 3, {"reason": why}
    feed, feed_why = build_feed(cfg, policy)
    if feed is None:
        say("STOPPED: %s" % feed_why)
        return 3, {"reason": feed_why}
    ident = FaceIdentifier(
        gallery, recogniser,
        min_conf=float(cfg.get("camera.min_conf", 0.6)),
        match_min=float(cfg.get("camera.identity_min", SFACE_COSINE_SAME)),
        owner=owner_label(cfg))
    rig = vr.Rig(cam.FeedSource(feed), detector, feed.lens,
                 cam.thresholds_from_config(cfg), identifier=ident,
                 head=cam.head_from_config(cfg))
    say("Look at the camera and at your screen, as you normally would.")
    try:
        result = rig.run(frames=int(args.frames))
    finally:
        feed.close()
    payload = result.to_dict()
    vr.assert_numbers_only(payload)
    for line in result.lines():
        say(line)
    return (0 if result.ok else 1), payload


def do_backup(gallery: FaceGallery, dest: str, say) -> tuple:
    out = fe.backup(gallery, dest)
    say("backup     %s" % out["dest"])
    say("           %d generation(s) copied, %d verified by reading them "
        "back, %d embeddings" % (out["copied"], out["verified"],
                                 out["samples"]))
    # WHAT THE DESTINATION HOLDS, not only what was copied into it. A backup
    # directory is never reconciled with the source, so after a --rollback
    # the discarded generation is still there and is still the NEWEST -- and
    # the newest is the only thing --restore looks at. Reporting the copy
    # count alone hid that entirely.
    say("           it now holds generation(s) %s; newest is %d with %d "
        "embeddings, and that is what --restore would take"
        % (", ".join(str(g) for g in out["dest_generations"]) or "-",
           out["dest_newest"], out["dest_newest_samples"]))
    if out["not_in_live"]:
        say("           NOTE generation(s) %s are in this directory but NOT "
            "in the live gallery any more (a --rollback, or an older "
            "backup). Nothing here deletes them for you."
            % ", ".join(str(g) for g in out["not_in_live"]))
    for line in out["failed"]:
        say("           FAILED %s" % line)
    if not out["ok"]:
        say("           A backup nobody has read back is not a backup; that "
            "is the shape the voiceprint's copy had on 2026-09-02, and it "
            "held only the fixtures.")
    say("")
    say("This is a copy on whatever disk you pointed it at. If that is the "
        "same disk, you have a copy and not a backup.")
    return (0 if out["ok"] else 1), out


def do_restore(gallery: FaceGallery, src: str, say) -> tuple:
    gallery.load()
    out = fe.restore(gallery, src, reason="face_enrol --restore %s" % src)
    if not out["restored"]:
        say("STOPPED: %s" % out["reason"])
        return 3, out
    # Which generation it TOOK and what it displaced, because a backup
    # directory is never reconciled with the live gallery: after a --rollback
    # the discarded generation is still the newest thing in the backup, and
    # the newest is what this takes. These two numbers are how he sees that
    # before it matters.
    say("restored   generation %d from %s (%d embeddings there) over a live "
        "gallery that held %d"
        % (out["src_generation"], out["src"], out["src_samples"],
           out["live_before"]))
    say("           written as generation %d, %d embeddings in the gallery "
        "now" % (out["generation"], out["samples"]))
    say("Nothing was overwritten: a restore is a new generation, so a "
        "restore of the wrong thing is one --rollback away.")
    return 0, out


def do_rollback(gallery: FaceGallery, say) -> tuple:
    before = gallery.generations()
    gen = gallery.rollback()
    if not gen:
        say("STOPPED: there is no earlier generation to roll back to "
            "(%d on disk)." % len(before))
        return 3, {"generations": len(before)}
    say("rolled back to generation %d, %d embeddings" % (gen, gallery.total()))
    return 0, {"generation": gen, "total": gallery.total()}


def do_delete(cfg, gallery: FaceGallery, args, say) -> tuple:
    """Destroy the biometric data. Every generation, and the crashed-save
    leftovers that are invisible to the generation pattern."""
    gens = gallery.generations()
    if not gens and not gallery.root.exists():
        say("Nothing to delete at %s" % gallery.root)
        return 3, {"removed": 0}
    if not args.yes:
        say("About to permanently destroy %d generation(s) of face "
            "embeddings at %s." % (len(gens), gallery.root))
        say("This cannot be undone and there is no backup unless you made "
            "one with --backup.")
        try:
            answer = input('Type "delete" to confirm: ').strip().lower()
        except EOFError:
            answer = ""
        if answer != "delete":
            say("Not deleted.")
            return 1, {"removed": 0}
    removed = gallery.purge()
    left = [p.name for p in sorted(gallery.root.iterdir())] \
        if gallery.root.is_dir() else []
    say("deleted    %d file(s); %d left in the directory (%s)"
        % (removed, len(left), ", ".join(left) or "-"))
    turned_off = cfg.set("camera.identity", False) is not False
    say("           camera.identity set to false: %s" % turned_off)
    # SAY WHAT WAS ACTUALLY DESTROYED. The old line here claimed "your face
    # is no longer on this disk", which contradicted this command's own
    # confirm prompt three lines above ("no backup unless you made one with
    # --backup") -- a --backup copy survives this fully loadable -- and
    # over-claimed the mechanism as well: the files are overwritten and
    # unlinked, which puts the bytes beyond the filesystem, not beyond the
    # device (an SSD controller may still hold the old blocks).
    say("The gallery at %s is gone: %d file(s) overwritten and unlinked."
        % (gallery.root, removed))
    say("Copies you made with --backup are NOT touched by this -- if you "
        "made one, delete it yourself. Jarvis must restart for the flag to "
        "take effect.")
    return (0 if not left else 1), {"removed": removed, "left": len(left),
                                    "identity_off": bool(turned_off)}


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true",
                    help="what is enrolled, in numbers. Opens no device")
    ap.add_argument("--verify", action="store_true",
                    help="run the live camera against the stored gallery")
    ap.add_argument("--backup", metavar="DIR",
                    help="copy every generation there and read each back")
    ap.add_argument("--restore", metavar="DIR",
                    help="load a backup as a NEW generation; nothing is "
                         "overwritten")
    ap.add_argument("--rollback", action="store_true",
                    help="delete the newest generation and load the one "
                         "before it")
    ap.add_argument("--delete", action="store_true",
                    help="destroy every generation of your face embeddings")
    ap.add_argument("--reset", action="store_true",
                    help="destroy the older generations AFTER this enrolment "
                         "is captured and saved. Asks for confirmation, and "
                         "destroys nothing if the run fails a check")
    ap.add_argument("--append", action="store_true",
                    help="add to the existing pool instead of starting "
                         "fresh. The checks then judge the MERGED pool, "
                         "which is what gets saved")
    ap.add_argument("--enable-identity", action="store_true",
                    help="set camera.identity true, the gate on writing "
                         "anything about your face down")
    ap.add_argument("--frames", type=int, default=120,
                    help="frames for --verify (default 120)")
    ap.add_argument("--frames-per-station", type=int, default=240,
                    help="frame budget per station before moving on")
    ap.add_argument("--gap-s", type=float, default=0.35,
                    help="minimum seconds between two accepted samples, so "
                         "one pose cannot become eight near-duplicates")
    ap.add_argument("--min-face-px", type=float, default=fe.MIN_FACE_PX,
                    help="reject a face smaller than this in capture pixels "
                         "(SFace's input is 112x112)")
    ap.add_argument("--min-sharpness", type=float, default=fe.MIN_SHARPNESS,
                    help="reject a blurrier sample than this. PROVISIONAL: "
                         "nobody has measured it on your face, and the "
                         "report prints the whole distribution")
    ap.add_argument("--auto", action="store_true",
                    help="do not wait for Enter between stations")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="save even though the new pool is smaller than what "
                         "is on disk")
    ap.add_argument("--force", action="store_true",
                    help="save even though a check failed. Recorded in the "
                         "generation's provenance")
    ap.add_argument("--yes", action="store_true",
                    help="do not ask for confirmation on --delete or --reset")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable, and just as pixel-free")
    args = ap.parse_args(argv)

    say = (lambda *a: None) if args.json else print
    say(BANNER)
    say("")

    cfg = AssistantConfig.load()
    gallery = open_gallery()
    report: dict = {"banner": BANNER, "gallery": str(gallery.root)}

    if args.status:
        code, payload = do_status(cfg, gallery, say)
    elif args.backup:
        code, payload = do_backup(gallery, args.backup, say)
    elif args.restore:
        code, payload = do_restore(gallery, args.restore, say)
    elif args.rollback:
        code, payload = do_rollback(gallery, say)
    elif args.delete:
        code, payload = do_delete(cfg, gallery, args, say)
    else:
        policy = SensingPolicy(cfg=cfg)
        report["sensing"] = policy.status()
        say("sensing    camera=%s reason=%s curfew=%s failsafe=%s"
            % (report["sensing"]["camera"], report["sensing"]["reason"] or "-",
               report["sensing"]["curfew"] or "-",
               report["sensing"]["failsafe"]))
        if args.verify:
            code, payload = do_verify(cfg, policy, gallery, args, say)
        else:
            code, payload = do_enrol(cfg, policy, gallery, args, say)

    report["result"] = payload
    report["exit_code"] = code
    if args.json:
        vr.assert_numbers_only(report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
