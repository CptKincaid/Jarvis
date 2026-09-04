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
    ~/vss_env/bin/python scripts/face_enrol.py --pose "looking at my phone"
    ~/vss_env/bin/python scripts/face_enrol.py --label heather # somebody else
    ~/vss_env/bin/python scripts/face_enrol.py --delete --label heather

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

A NOTE ON EVERY TAKE, AND TAKES YOU NAME YOURSELF. Each accepted sample
stores your own words for what you were doing -- the five default stations
carry one each ("looking at the lens", "looking at my screen", ...), and
``--pose "looking at my phone"`` adds a station of your own. The notes are
the only thing that can answer the question a bad match actually raises:
``--status`` groups the pool's own cohesion by note, worst first, so "it did
not know me just then" gets "your looking-at-my-phone takes cohere least"
instead of a shrug. Repeat ``--pose`` for as many takes as you want;
``--pose-samples`` sets how many frames each one wants.

THE GAP, RATHER THAN THE LIST AGAIN. ``pose_spread`` counts abs(yaw), so it
cannot see that every sample of both your enrolment and your verification
carried a POSITIVE yaw -- you have no coverage on the other side at all. The
report and ``--status`` now count the two sides separately and say which one
is empty, and the next run asks for exactly the missing station instead of
reading the same five instructions back at you. A gallery that records no
angles (generation 1 does not) supports no inference, so that case runs the
five stations, which were chosen against your measured geometry.

SOMEBODY ELSE, AND THEIR CONSENT. ``--label heather`` enrols a second person.
It stores THEIR biometric data, which is theirs to agree to and not yours, so
the run prints what is stored, what it can never do (a recognised face
commands nothing and unlocks nothing -- see ``owner_label``), and how to
delete it, and then stops until THEY type their own name. ``--yes`` is your
flag and cannot give it; ``--json`` cannot either, because the text would be
invisible. ``--delete --label heather`` removes that one person from EVERY
generation -- not just the newest -- and leaves everybody else's alone.

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
from jarvis.eye import FaceIdentifier                     # noqa: E402
from jarvis.facegallery import (SFACE_COSINE_SAME,        # noqa: E402
                                FaceGallery, default_gallery, label_ok)
from jarvis.sensing import SensingPolicy                   # noqa: E402

BANNER = (
    "face enrolment -- numbers only. No frame is displayed, saved, described\n"
    "or written; every line below is a count, an angle, a size, a score or a\n"
    "millisecond, and the only thing that reaches the disk is the\n"
    "embedding.")

BACKUP_WARNING = (
    "NOTE ON BACKUP: the generations below all live in the SAME directory on\n"
    "the SAME disk. They make a bad write recoverable; they do not survive a\n"
    "disk failure, and nothing under ~/.aiws_trainer is backed up on this\n"
    "box (restic is installed, no repository exists). Run --backup to a\n"
    "different disk if this enrolment is worth keeping.")


# --------------------------------------------------------------- the seams
def open_gallery(cfg=None) -> FaceGallery:
    """The user's gallery FOR THE ACTIVE MODEL. ``PATHS.FACE_GALLERY`` honours
    ``JARVIS_FACE_GALLERY``, which tests/conftest.py forces into a throwaway
    directory -- so nothing in the suite can reach the real one.

    The model matters here more than anywhere: enrolling ArcFace vectors into
    a gallery labelled SFace would produce a store nothing could ever read."""
    backend = "" if cfg is None else cam.face_backend_from_config(cfg)
    return default_gallery(backend=backend)


def build_models(cfg):
    """``(detector, recogniser, reason)``. A DELEGATE -- see fe.build_models.

    The body moved into the library when the in-app run became a second
    caller, so that "which models does enrolment load" has one answer. The
    name is kept here because it is what this script's own tests drive.
    """
    return fe.build_models(cfg)


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
            lens=lens, present=lambda: cam.device_present(device),
            device=device), why
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def owner_label(cfg) -> str:
    """HIS label -- the one name that means "the owner" anywhere downstream.

    THE RULING THAT CHANGED, AND THE ONE THAT DID NOT. He asked on
    2026-09-02 to enrol other people ("so i can enroll others"), which
    reverses the earlier him-only rule; ``--label`` is that, and it takes the
    other person's typed consent. What has NOT changed is the governing rule:
    identity may REMOVE capability or ADD a name, and must NEVER GRANT
    capability the existing gates do not already grant.

    This function is where that rule is anchored. ``owner`` is read from HIS
    CONFIG and never from the gallery, so the set of enrolled names can grow
    without the set of privileged names growing by one. ``jarvis/eye.py``'s
    wake fusion asks ``eye.identity in ("", owner)``: an unknown face is
    exactly today's behaviour, HIS face is exactly today's behaviour, and any
    other enrolled name WITHHOLDS a promotion an anonymous face would have
    got. Enrolling Heather can only ever make Jarvis more careful. Pinned by
    tests/test_faceenrol_notes.py::test_enrolling_a_second_person_can_only_TAKE_a_promotion_AWAY.
    """
    # A DELEGATE. It used to derive the label here, which was the third of
    # three copies -- and the one that kept every character isalnum() liked,
    # including non-ASCII, while the gallery stores under
    # ^[a-z0-9][a-z0-9_-]{0,30}$. A user.name of "Jose" with an accent
    # produced a label the gallery would not take. identity.slug fixes that
    # for all three callers at once.
    from jarvis.identity import owner_label as _one_owner_label
    return _one_owner_label(cfg)


def target_label(cfg, args) -> tuple:
    """``(label, why)`` -- who this run is about. ``why`` is "" when it is
    fine and the refusal text when it is not.

    The label goes into an npz key, a log line and a report he pastes, so it
    is checked against ``facegallery``'s own pattern BEFORE anything is
    opened: a name that cannot be stored must cost the run at the argument,
    not after a minute in front of the camera.

    HIS OWN LABEL IS CHECKED TOO, and it used to be the one that was not.
    ``owner_label`` derives it from ``user.name``, keeping any character
    ``isalnum()`` likes -- which includes non-ASCII -- while the gallery
    stores under ``^[a-z0-9][a-z0-9_-]{0,30}$``. A ``user.name`` of "Jose"
    with an accent, or one over 31 characters, produced a label
    ``gallery.add`` refuses on EVERY sample: ``EnrolmentSession.offer``
    catches the ValueError and records "gallery refused the embedding", so
    the run spent his minute in front of the camera, captured nothing, and
    failed the samples check at the end for a reason the config could have
    given at the start. The voice path already checked (``enrol_answer``
    calls ``label_ok`` and answers "That isn't a name I can store, sir"); the
    two entry points now agree."""
    want = str(getattr(args, "label", "") or "").strip().lower()
    if not want:
        mine = owner_label(cfg)
        if not label_ok(mine):
            return "", ("your own name in the config does not make a usable "
                        "label (%r from user.name %r): lowercase letters, "
                        "digits, - and _ only, up to 31 characters. Set "
                        "user.name in ~/.config/jarvis/assistant.json, or "
                        "pass --label."
                        % (mine, str(cfg.get("user.name", "") or "")))
        return mine, ""
    if not label_ok(want):
        return "", ("%r is not a usable label: lowercase letters, digits, - "
                    "and _ only, up to 31 characters. It becomes a key in "
                    "the gallery file and a word in every report."
                    % str(getattr(args, "label", "")))
    return want, ""


CONSENT_LINES = (
    "CONSENT -- this is %(who)s's data, not yours.",
    "",
    "Enrolling %(who)s stores a measurement of %(who)s's FACE: 128 numbers",
    "per take, in %(root)s, at 0600 in a 0700",
    "directory. No photograph, no video and no crop is stored, nothing is",
    "displayed, and nothing leaves this machine. It cannot be re-issued if",
    "it leaks, which is why it is treated like the voiceprint and not like",
    "a setting.",
    "",
    "What it is FOR: Jarvis can say %(who)s is here, and can keep something",
    "private when somebody else is in the room. What it never does: a",
    "recognised face cannot command Jarvis, cannot unlock anything and",
    "cannot pass the voice gate. Recognising a face may only ever make",
    "Jarvis do LESS, never more.",
    "",
    "Deleting it, at any time, and it takes about a second:",
    "    %(python)s %(script)s --delete --label %(who)s",
    "which destroys every generation that holds %(who)s -- including the",
    "older ones -- and leaves everybody else's alone.",
    "",
    "%(who)s must type their own name below. Nobody may type it for them,",
    "and --yes cannot do it either: it is his flag, and this is not his",
    "consent to give.",
)


def _isatty(stream) -> bool:
    """True only when ``stream`` is really a terminal.

    A stream that cannot answer is NOT a terminal: this gates biometric
    consent, so the unknown case has to fail closed."""
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001 - a stream that cannot say is not a tty
        return False


def consent(label: str, owner: str, root, say, args) -> tuple:
    """``(ok, how)``. Enrolling somebody else takes THEIR agreement.

    NOT A COMMENT AND NOT A README LINE. Storing a second person's biometric
    data without them knowing is the failure this flow exists to prevent, so
    the words are printed, the person is named, and the run stops until they
    type their own name.

    ``--json`` CANNOT GIVE IT. In JSON mode ``say`` is a no-op and stdout is
    a machine-readable document, so the consent text is invisible and the
    prompt would land in the middle of it -- a consent nobody could read is
    not a consent, and a pipe is exactly how "enrol whoever is in frame"
    would get automated.

    AND NEITHER CAN A PIPE, which is the same sentence and used to be only
    half enforced. Blocking --json closed the flag and left the mechanism it
    was named after wide open: ``echo heather | face_enrol.py --label heather
    --auto --yes`` satisfied this prompt with nobody at the keyboard, and the
    run then wrote "consent typed at the keyboard" into the generation's
    provenance -- a FALSE ATTESTATION on disk, which is worse than no record,
    because the provenance string is the only durable thing that claims the
    consent happened at all. So both ends are checked: stdin must be a
    terminal (somebody typed it) and stdout must be a terminal (they could
    read what they were agreeing to). Redirecting either one is refused with
    the same sentence --json gets.
    """
    if label == owner:
        return True, "owner"
    fields = {"who": label, "root": str(root), "python": sys.executable,
              "script": os.path.abspath(__file__)}
    if args.json:
        return False, ("consent for %r cannot be taken through --json: the "
                       "person being enrolled has to read what is stored "
                       "and type their own name" % label)
    if getattr(args, "auto", False):
        # --auto removes the Enter between stations. For his own face that is
        # a convenience; for somebody else's it removes the only evidence,
        # station by station, that the person is still there and still
        # willing -- a capture nobody has to press a key for is a capture
        # nobody has to be present for.
        return False, ("--auto cannot be used to enrol %r: it removes the "
                       "key press between stations, and somebody else's "
                       "enrolment is the one that needs them at the "
                       "keyboard throughout" % label)
    if not (_isatty(sys.stdin) and _isatty(sys.stdout)):
        return False, ("consent for %r cannot be taken through a pipe: the "
                       "person being enrolled has to read what is stored "
                       "and type their own name, at a terminal. Run this "
                       "without redirecting stdin or stdout." % label)
    say("")
    for line in CONSENT_LINES:
        say(line % fields)
    try:
        answer = input('Type "%s" to agree: ' % label).strip().lower()
    except EOFError:
        answer = ""
    if answer != label:
        return False, ("consent was not given for %r -- nothing was "
                       "captured and nothing was written" % label)
    return True, "typed"


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
        one = FaceGallery(root=gallery.root, model=gallery.model)
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
    payload["notes"] = []
    payload["coverage"] = {}
    # PER LABEL, KEYED BY LABEL. These five used to be written flat onto the
    # payload inside this loop, so with two people enrolled the last label
    # round the loop overwrote the first and --json reported ONE pool's
    # numbers as the gallery's: measured on a two-person store, a healthy
    # cohesion_min 0.998 printed over a gallery whose weakest pool was 0.466.
    # The text half was right all along, which is exactly why nobody saw it.
    # The gallery-wide keys below are the MIN across labels, which is the
    # only aggregate that means anything for a "worst pool" number.
    payload["by_label"] = {}
    for label in gallery.labels():
        embs = gallery.embeddings(label)
        takes = gallery.takes(label)
        pairs = fe.pairwise_cosines(embs)
        coh = fe.cohesion(embs)
        one = {"pairs": len(pairs),
               "cos_min": min(pairs) if pairs else 0.0,
               "cos_p50": vr._pct(pairs, 0.5),
               "cos_max": max(pairs) if pairs else 0.0,
               "cohesion_min": min(coh) if coh else 0.0}
        payload["by_label"][label] = one
        say("  %-10s %d embeddings, %d pairs, cosine min %.3f p50 %.3f "
            "max %.3f, worst cohesion %.3f"
            % (label, len(embs), len(pairs),
               one["cos_min"], one["cos_p50"], one["cos_max"],
               one["cohesion_min"]))
        # WHAT EACH POSE IS WORTH. The notes exist to answer one question --
        # "which pose is letting me down" -- and --status is where he asks
        # it, because it is the mode that needs no camera and no minute of
        # his time.
        cov = fe.coverage(takes)
        payload["coverage"][label] = dict(cov)
        for line in fe.coverage_lines(cov):
            say("  " + line)
        rows = fe.note_rows(embs, takes)
        if rows:
            say("  by take    weakest first -- this is where to add takes")
            for row in rows:
                say("  " + row.line())
            payload["notes"].extend([label] + list(row.as_tuple())
                                    for row in rows)
        weakest = fe.weakest_note(rows)
        if weakest:
            mine = label == owner_label(cfg)
            say("  WEAKEST    %s %r takes cohere least with the rest of %s "
                "pool -- that is where to add takes, not to the pose that "
                "is already strong."
                % ("your" if mine else "%s's" % label, weakest,
                   "your" if mine else "their"))
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

    owner = owner_label(cfg)
    label, why = target_label(cfg, args)
    if why:
        say("STOPPED: %s" % why)
        return 1, {"reason": why}
    # THE PHASE GATE IS SPLIT IN TWO AROUND THE CONSENT STEP, ON PURPOSE.
    # The READ half runs first, so a person is never asked to agree to a
    # capture that was going to stop anyway; the WRITE half runs after, so a
    # consent that was REFUSED cannot have left the switch that says faces
    # may be written down flipped on behind it. Neither ordering gives both
    # properties on its own.
    identity_on = bool(cfg.get("camera.identity", False))
    if not identity_on and not args.enable_identity:
        say("STOPPED: camera.identity is false, and it is the phase gate "
            "for writing anything about your face down at all.")
        say("Re-run with --enable-identity to turn it on and enrol, or "
            "set camera.identity true in ~/.config/jarvis/assistant.json.")
        return 1, {"reason": "camera.identity is false"}
    ok, how = consent(label, owner, gallery.root, say, args)
    if not ok:
        say("STOPPED: %s" % how)
        return 1, {"reason": how, "label": label}
    if not identity_on:
        if cfg.set("camera.identity", True) is False:
            say("STOPPED: camera.identity could not be written to the config.")
            return 1, {"reason": "camera.identity could not be set"}
        say("camera.identity is now true (Jarvis must restart to read it).")

    # backend= is not optional here. Without it ``probe`` resolves the
    # default pair, so a box configured for "opencv" is told the InsightFace
    # weights are missing and enrolment stops over a file it was never going
    # to load. It is the same dropped-model mistake as _generation_holds'.
    probe = facedetect.probe(model_dir=str(cfg.get("camera.model_dir", "")
                                           or "") or None,
                             backend=cam.face_backend_from_config(cfg))
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
            # ...and it IS ignored: this flag used to survive the sentence,
            # so the stored takes were carried into the new generation and
            # the old ones were then destroyed -- the pool he asked to
            # replace came forward in full (F14, reproduced 2026-09-03).
            args.append = False
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

    # THE GALLERY IS ALWAYS LOADED FIRST, AND THAT IS NOT AN --append.
    # save() writes the WHOLE in-memory pool as the next generation, so a
    # run that started from an empty object would write a generation holding
    # this label alone -- and every other enrolled person would silently stop
    # being recognised the moment it landed, while still sitting in the
    # generation underneath. Loading first and then dropping THIS label (for
    # anything but --append) keeps the old meaning of a plain re-run -- a
    # fresh pool for him -- while leaving everybody else exactly where they
    # were.
    gallery.load()
    others = [x for x in gallery.labels() if x != label]
    stored_takes = gallery.takes(label)
    if args.append:
        if gallery.count(label):
            say("--append: starting from generation %d, %d embeddings under "
                "%r. The checks below judge the MERGED pool -- the %d "
                "already stored plus whatever this run adds -- because that "
                "is what gets saved."
                % (gallery.loaded_generation, gallery.count(label), label,
                   gallery.count(label)))
    else:
        gallery.forget(label)
    if others:
        say("keeping    %d other label(s) untouched: %s"
            % (len(others), ", ".join(others)))

    # THE GAP IS RELATIVE TO WHAT SURVIVES THIS RUN, not to what is on disk.
    # A plain re-run and --reset both REPLACE this label's pool, so the new
    # pool has to stand on its own and the plan is the full script; only
    # --append carries the stored takes forward, and only there does "the
    # station you are missing" mean anything. Asking a replacing run for the
    # gap alone would write a three-sample gallery over a thirteen-sample
    # one and call it coverage.
    plan_takes = stored_takes if args.append else []
    plan, plan_why = fe.choose_plan(plan_takes, poses=args.pose,
                                    pose_samples=int(args.pose_samples),
                                    mode=args.plan)
    if not plan:
        say("STOPPED: %s. Name a take with --pose \"looking at my phone\", "
            "or --plan full to run the five stations again." % plan_why)
        return 1, {"reason": plan_why, "label": label}

    session = fe.EnrolmentSession(gallery, label, feed.lens, detector,
                                  recogniser, limits,
                                  head=cam.head_from_config(cfg))
    say("")
    say("Enrolling %r. %d stations, %d samples wanted -- %s."
        % (label, len(plan), sum(st.samples for st in plan), plan_why))
    # The coverage of what is STORED, said before the run whether or not it
    # is carried forward: "you have never given the other side of your face"
    # is worth knowing when you are about to sit down for a minute.
    if stored_takes:
        for line in fe.coverage_lines(fe.coverage(stored_takes)):
            head, rest = (line.split(None, 1) + [""])[:2]
            say("stored     %s" % rest if head == "coverage" else line)
    # NAMING POSES IS ADDITIVE BY INTENT -- "more ways for me to be
    # recognised" -- and a plain run REPLACES. Said BEFORE the capture,
    # because learning it afterwards costs the minute plus a refusal he
    # could not have predicted: a two-pose run is six takes, under the
    # eight-sample floor, so it captures, fails and saves nothing.
    wanted = sum(st.samples for st in plan)
    if args.pose and not args.append and stored_takes:
        say("WARNING    these %d take(s) will REPLACE the %d already stored "
            "under %r. Add --append to keep what is there."
            % (wanted, len(stored_takes), label))
        if wanted < fe.MIN_SAMPLES:
            say("           %d is under the %d-sample floor, so this run "
                "would be refused after the capture. --append is almost "
                "certainly what you want."
                % (wanted, fe.MIN_SAMPLES))
    say("Sit where you normally sit. Nothing you see is shown to anybody, "
        "because nothing is shown at all.")
    def wait(prompt):
        input(prompt)

    if args.auto:
        wait = None
    try:
        rep, _run = fe.run_enrolment(
            session, cam.FeedSource(feed), plan=plan, say=say, wait=wait,
            frames_per_station=int(args.frames_per_station),
            gap_s=float(args.gap_s),
            identity_min=float(cfg.get("camera.identity_min",
                                       SFACE_COSINE_SAME)))
    finally:
        feed.close()

    # THE VERDICT GUARD IS NO LONGER WRITTEN OUT HERE. It moved into
    # fe.save_enrolment so that this script and the in-app run
    # (jarvis/enrolrun.py) are guarded by the same code rather than by two
    # copies of the same sentence. --force is passed from here and from
    # nowhere else; the window has no way to reach it.
    saved = fe.save_enrolment(
        gallery, rep,
        reason="face_enrol %s%s%s"
        % (label,
           " (consent typed at the keyboard)" if how == "typed"
           else "",
           " --force" if not rep.ok else ""),
        allow_shrink=bool(args.allow_shrink), force=bool(args.force),
        superseded=superseded)
    removed = saved["removed"]
    if saved["error"]:
        say("")
        say("NOT SAVED: %s" % saved["error"])
    payload = rep.to_dict()
    payload["reset_removed"] = removed
    payload["consent"] = how
    payload["plan"] = plan_why
    payload["kept_labels"] = list(others)
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
    bar_why = cam.identity_min_warning(cfg)
    if bar_why:
        say("NOTE: %s" % bar_why)
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
        # rollback() refuses BEFORE it shreds when no older generation
        # parses, so a 0 here means the disk is exactly as it was -- and
        # the line says so, instead of counting files that may be unreadable.
        after = gallery.generations()
        say("STOPPED: there is no readable earlier generation to roll back "
            "to (%d on disk, %s)." % (len(before),
                                     "nothing touched" if after == before
                                     else "%d left" % len(after)))
        return 3, {"generations": len(after), "touched": after != before}
    say("rolled back to generation %d, %d embeddings" % (gen, gallery.total()))
    return 0, {"generation": gen, "total": gallery.total()}


def do_delete_label(gallery: FaceGallery, label: str, args, say) -> tuple:
    """Destroy ONE person's embeddings, everywhere, and leave the rest.

    THE HALF OF CONSENT THAT IS NOT THE PROMPT. Agreeing to be enrolled is
    only meaningful if withdrawing is one command, so this is one command --
    and it has to mean it: ``forget()`` plus a save would leave her in every
    older generation, one --rollback from coming back and still lying on the
    disk as 128 floats a take.

    WHAT ``purge_label`` PROMISES, and therefore what this may print: no file
    is destroyed unless the embeddings it held, MINUS HERS, have been read
    back from the new generation on disk -- by name and by SAMPLE COUNT, off
    the disk, not from a return value. Anything that cannot be proved that way
    is left alone and named here, and the "verified" line is withheld.

    THE THREE THINGS IT WILL NOT DO, all the same rule:

    * destroy a file it could not read. ``_read`` refuses a FORMAT number it
      does not know -- that is the point of the check -- so on the day the
      format changes every existing generation is "unreadable", and a
      ``--delete --label somebody-who-was-never-enrolled`` that destroyed
      them would take the whole gallery (measured 2026-09-03: 13 embeddings,
      ``_format`` bumped by one, empty directory, exit 0).
    * destroy one it could not rewrite -- another model's generation, his
      SFace enrolment with an ArcFace build live.
    * destroy one holding somebody the new generation does not carry at full
      count. A label is not an enrolment: the version of this check that
      asked "is he in the new file?" destroyed a generation holding ten of
      his takes on the strength of one holding three.

    In all three she is then still on the disk, so this says that too rather
    than "verified": ``complete`` comes back False and the command exits
    non-zero. ``--delete`` with no ``--label`` is the command that means
    everything.

    ``camera.identity`` is NOT touched here. It is the switch on his own
    face being written down at all; removing somebody else must not turn his
    recognition off behind his back."""
    gallery.load()
    out = gallery.purge_label(label, reason="face_enrol --delete --label %s"
                              % label)
    unreadable = list(out.get("unreadable") or ())
    foreign = list(out.get("foreign") or ())
    foreign_models = (", ".join(out.get("foreign_models") or ())
                      or "another model")
    not_carried = list(out.get("not_carried") or ())
    # WHO IS ENROLLED, READ OFF THE DISK. This used to print
    # ``gallery.labels()``, which is what this OBJECT loaded -- and a gallery
    # for one model loads NOTHING from another model's store, so the line said
    # "(labels: -)" over a full enrolment sitting right there.
    on_disk = ", ".join(out.get("labels_on_disk") or ()) or "-"

    def _unreadable_lines():
        say("           %d generation(s) could NOT be read and were left "
            "alone: %s. Nothing can say whether they hold %r, so nothing "
            "here may claim they do not."
            % (len(unreadable), ", ".join(str(g) for g in unreadable), label))
        say("           Run --status, then --rollback to drop a bad newest "
            "generation, or --delete with no --label to destroy everything.")

    if not out["generations_with"]:
        say("Nothing enrolled under %r at %s (labels: %s)"
            % (label, gallery.root, on_disk))
        if out.get("tmp_removed"):
            say("           %d crashed-save leftover(s) destroyed as well: "
                "a .tmp holds a whole pool and no name can filter it."
                % out["tmp_removed"])
        if unreadable:
            _unreadable_lines()
            return 1, out
        return 3, out
    if out["reason"]:
        say("STOPPED: %s" % out["reason"])
        say("Nothing was destroyed -- what is left could not be written, or "
            "could not be read back off the disk afterwards, and deleting one "
            "person may not cost another person's enrolment.")
        return 1, out
    if out["removed"]:
        say("deleted    %r from %d of the %d generation(s) that held her; "
            "%d file(s) overwritten and unlinked"
            % (label, out["removed"], len(out["generations_with"]),
               out["removed"]))
    else:
        say("deleted    nothing: none of the %d generation(s) holding %r "
            "could be destroyed without costing somebody else theirs."
            % (len(out["generations_with"]), label))
    if out.get("tmp_removed"):
        say("           %d crashed-save leftover(s) destroyed as well: a "
            ".tmp holds a whole pool and no name can filter it."
            % out["tmp_removed"])
    # WHAT IS LEFT IS A STATEMENT ABOUT THE DISK, not about whether this
    # command happened to write a generation. When every file holding her held
    # ONLY her there is nobody to carry forward and nothing is written -- and
    # the old wording then printed "nothing is left; the gallery is empty"
    # over everybody else's generations.
    if out["left"]:
        if out["generation"]:
            say("           what is left is generation %d: %d embeddings over "
                "%s" % (out["generation"], out["left"],
                        ", ".join(out["labels_left"]) or "nobody"))
        else:
            say("           nothing needed rewriting: %d embeddings over %s "
                "are still on disk in generation %d"
                % (out["left"], ", ".join(out["labels_left"]) or "nobody",
                   out.get("loaded") or 0))
    elif out.get("labels_on_disk"):
        say("           nothing this gallery can read is left, and %s are "
            "still on this disk under another model." % on_disk)
    else:
        say("           nothing is left; the gallery is empty.")
        # camera.identity stays as it is, and that is said rather than left
        # to be discovered: this command's job is one person, and the flag
        # is the switch on the whole feature. A --delete with no --label is
        # the one that turns it off.
        say("           camera.identity is untouched -- run --delete with "
            "no --label to turn the feature off as well.")
    # Read it back rather than claim it, and read it back TWICE over. A delete
    # that reports success over a file that still parses with her in it is the
    # whole failure mode. ``purge_label`` re-inventories the disk itself --
    # raw, so it can see her in a file this gallery's model cannot load, which
    # is the case ``_generation_holds`` cannot answer at all -- and this walks
    # the generations independently on top of that. The tmps are counted here
    # too and not only the generations, because a .tmp holds a whole pool.
    back = FaceGallery(root=gallery.root, model=gallery.model)
    back.load()
    still = sorted(set(out.get("still_holding") or ())
                   | {g for g in back.generations()
                      if _generation_holds(back.root, g, label, back.model)})
    left_tmps = back.leftovers()
    if foreign:
        say("           NOT VERIFIED: generation(s) %s were written by %s and "
            "still hold %r. This build writes %s and cannot rewrite them, so "
            "destroying them would cost everybody else in them their "
            "enrolment. They were LEFT ALONE."
            % (", ".join(str(g) for g in foreign), foreign_models, label,
               gallery.model))
        say("           Set camera.face_backend back to the model that wrote "
            "them and run this again, or --delete with no --label to destroy "
            "everything.")
    if not_carried:
        say("           NOT VERIFIED: generation(s) %s still hold %r and were "
            "LEFT ALONE: somebody else in them is not in the new generation "
            "with at least the samples they had, so destroying them would "
            "cost that person part of their enrolment."
            % (", ".join(str(g) for g in not_carried), label))
        say("           --delete with no --label is the command that destroys "
            "everything.")
    if unreadable:
        # The one thing this command may not do is print "verified" over a
        # file nothing could open.
        _unreadable_lines()
    if still and not (foreign or not_carried):
        say("           FAILED: generation(s) %s still hold %r"
            % (", ".join(str(g) for g in still), label))
    if left_tmps:
        say("           FAILED: %s survived, and a .tmp holds a whole pool"
            % ", ".join(left_tmps))
    if still or left_tmps or unreadable or not out.get("complete"):
        return 1, out
    say("           verified: no generation on disk holds %r any more."
        % label)
    say("Copies you made with --backup are NOT touched by this -- if you "
        "made one, delete it yourself. The bytes here were overwritten and "
        "unlinked, which puts them beyond the filesystem, not beyond the "
        "device.")
    return 0, out


def _generation_holds(root, generation: int, label: str, model: str) -> bool:
    """Does this one generation, READ AS ``model``, still hold ``label``?

    ``model`` IS REQUIRED AND HAS NO DEFAULT, and that is the fix rather than
    an accident. This used to build ``FaceGallery(root=root)``, which takes
    the module default -- so it was an SFace gallery while its caller was
    ArcFace's. On a mixed store the cross-model refusal then made every
    generation answer False, the read-back saw nothing anywhere, and
    ``--delete --label`` printed "verified: no generation on disk holds
    <label> any more" over embeddings that were still there. A default
    argument is exactly how the model got dropped in the first place, so
    there is not one: a caller that cannot say which model it means has no
    business asking this question.
    """
    one = FaceGallery(root=root, model=model)
    return bool(one.load(generation=generation)) and label in one.labels()


def do_delete(cfg, gallery: FaceGallery, args, say) -> tuple:
    """Destroy the biometric data. Every generation, and the crashed-save
    leftovers that are invisible to the generation pattern.

    With ``--label`` it is one person instead of everybody; see
    ``do_delete_label``."""
    gens = gallery.generations()
    if not gens and not gallery.root.exists():
        say("Nothing to delete at %s" % gallery.root)
        return 3, {"removed": 0}
    if args.label:
        label, why = target_label(cfg, args)
        if why:
            say("STOPPED: %s" % why)
            return 1, {"reason": why}
        if not args.yes:
            say("About to permanently destroy every embedding of %r at %s, "
                "in every generation that holds them. Everybody else's are "
                "left alone." % (label, gallery.root))
            say("This cannot be undone and there is no backup unless you "
                "made one with --backup.")
            try:
                answer = input('Type "%s" to confirm: ' % label).strip().lower()
            except EOFError:
                answer = ""
            if answer != label:
                say("Not deleted.")
                return 1, {"removed": 0, "label": label}
        return do_delete_label(gallery, label, args, say)
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
def build_parser() -> argparse.ArgumentParser:
    """The arguments, as an object.

    Split out of ``main`` so the in-app entry point can PARSE the command
    line it hands him before he runs it (jarvis/enrolentry.py). A command
    Jarvis dictated that argparse then rejects is worse than no command, and
    a test can only catch that if there is something to hand it to."""
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
    ap.add_argument("--label", metavar="NAME",
                    help="who this is about. Defaults to you (user.name). "
                         "Enrolling anybody else takes THEIR typed consent, "
                         "which --yes and --json cannot give for them. With "
                         "--delete it removes that one person from every "
                         "generation and leaves everybody else's alone")
    ap.add_argument("--pose", action="append", default=[], metavar="TEXT",
                    help="a take in your own words -- \"looking at my "
                         "phone\", \"looking away\", \"with my glasses "
                         "off\". Repeatable; each one is a station, and the "
                         "words are stored beside every embedding it "
                         "produces so a later report can say which pose is "
                         "weak")
    ap.add_argument("--pose-samples", type=int, default=3,
                    help="samples per --pose station (default 3)")
    ap.add_argument("--plan", choices=("auto", "full", "missing"),
                    default="auto",
                    help="auto: run the stations your gallery is MISSING "
                         "when it records enough to say, else the five. "
                         "full: the five stations. missing: the gaps only, "
                         "and stop if there are none")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    say = (lambda *a: None) if args.json else print
    say(BANNER)
    say("")

    cfg = AssistantConfig.load()
    gallery = open_gallery(cfg)
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
