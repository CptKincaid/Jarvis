#!/usr/bin/env python3
"""How well each face model separates HIM from everything else -- in numbers.

    ~/vss_env/bin/python scripts/face_model_compare.py
    ~/vss_env/bin/python scripts/face_model_compare.py --json
    ~/vss_env/bin/python scripts/face_model_compare.py --profile-deg 20

WHY THIS EXISTS AND WHY IT IS YOURS TO RUN. On 2026-09-03 you said, of the
old models: *"he also is recognizing me less from the side angle"*. That is
an ACCURACY question, and accuracy needs your face. Nothing in this
repository is permitted to look at what your camera sees, so nobody working
on the code can measure it -- which is exactly why the instrument exists
instead of a claim. You run it, it prints numbers, the numbers decide.

WHAT IT MEASURES, all of it from vectors already on disk:

  * the INTRA-PERSON floor -- every enrolled take of one person against
    every other take of that same person. The lowest of those is the worst
    case the identity bar has to sit under, and it is the number that
    matters more than the average;
  * the same split by POSE, frontal against profile, using the head angle
    recorded with each take. This is the side-angle question, answered or
    honestly declined: a gallery whose takes carry no angle CANNOT answer it,
    and this says so rather than averaging over the ignorance;
  * every NON-MATCHING pair available -- other enrolled people, if there are
    any, and a synthetic out-of-distribution probe generated here, which is
    the only other "not him" evidence that exists without a second person;
  * the per-embedding COST of each model, on synthetic input;
  * a suggested identity bar, with the counts it rests on printed beside it
    so you can see how thin the evidence is.

WHAT IT DOES NOT DO, MECHANICALLY. It never opens the camera, never opens a
file that a lens produced, and never prints or writes an image or a path to
one. It reads ``gen-NNNNN.npz`` embeddings -- floats -- and the synthetic
frames it makes, it makes itself with numpy. Like
``scripts/camera_mode_probe.py``, the first thing ``main`` does is tokenize
this file and refuse to run if its own CODE names a call that could turn
data into a picture; ``tests/test_face_model_compare.py`` pins the same rule
from outside and pins that the two scripts' forbidden lists are identical.

THE TWO MODELS ARE NEVER COMPARED WITH EACH OTHER'S VECTORS. An SFace
128-vector and an ArcFace 512-vector are not weakly comparable; the cosine
between them measures nothing. Each model is reported in its own column
against its own gallery, and if only one of them has an enrolment then only
one of them gets numbers -- which is the true state of things after a swap
and before a re-enrolment.
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import statistics
import sys
import time
import tokenize
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import facegallery as fg                     # noqa: E402
from jarvis import faceinsight as fi                     # noqa: E402
from jarvis import facemodels as fm                      # noqa: E402
from jarvis.assistant_config import AssistantConfig      # noqa: E402
from jarvis.config import PATHS                          # noqa: E402
from jarvis.visionrig import assert_numbers_only         # noqa: E402

# The calls that would turn data into something a person could look at. The
# same list scripts/camera_mode_probe.py refuses on, deliberately identical
# so the two cannot drift; the test asserts they match.
PIXEL_CALLS = frozenset({
    "read", "retrieve", "imshow", "imwrite", "imencode", "imdecode",
    "tofile", "save", "frombuffer", "namedWindow", "waitKey",
})

# Beyond this many degrees off-axis a take counts as PROFILE. 15 is a
# starting split, not a finding -- it is a command-line flag precisely
# because the right place to cut depends on his own distribution and this
# script is what shows him that distribution.
PROFILE_DEG = 15.0
# Synthetic crops per family for the out-of-distribution probe.
OOD_N = 16


def self_check(path: str = __file__) -> list:
    """The names in this file's CODE that are on the forbidden list. Empty
    means the file may run. Strings and comments are skipped, which is why
    the docstring above can discuss the rule without tripping it."""
    hits = []
    source = Path(path).read_bytes()
    for tok in tokenize.tokenize(io.BytesIO(source).readline):
        if tok.type == tokenize.NAME and tok.string in PIXEL_CALLS:
            hits.append("%s line %d" % (tok.string, tok.start[0]))
    return hits


# ------------------------------------------------------------- statistics
def pct(values, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def spread(values) -> dict:
    """min / p05 / p50 / p95 / max / n, and zeros with n=0 when empty.

    n IS THE POINT. A floor computed from three pairs is not the same
    evidence as one computed from three hundred, and a table that prints the
    number without the count invites the reader to forget that.
    """
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0,
                "max": 0.0, "mean": 0.0}
    return {"n": len(vals), "min": min(vals), "p05": pct(vals, 0.05),
            "p50": pct(vals, 0.50), "p95": pct(vals, 0.95), "max": max(vals),
            "mean": statistics.mean(vals)}


def cosines(a_list, b_list=None) -> list:
    """All pairwise cosines within one list, or across two."""
    if b_list is None:
        return [fg.cosine(a, b) for a, b in itertools.combinations(a_list, 2)]
    return [fg.cosine(a, b) for a in a_list for b in b_list]


# ------------------------------------------------------------ the gallery
def pose_of(take, profile_deg: float) -> str:
    """"frontal", "profile" or "unrecorded" for one take.

    ``yaw_deg is None`` means NOTHING WAS RECORDED, which is not zero. His
    live generation carries no angles at all, and treating those as 0.0 would
    report a fully frontal gallery that was never measured -- turning "this
    cannot answer the side-angle question" into a confident wrong answer.
    """
    if take.yaw_deg is None:
        return "unrecorded"
    return "profile" if abs(float(take.yaw_deg)) > profile_deg else "frontal"


def one_model(root: Path, model: str, profile_deg: float) -> dict:
    """Everything measurable about one model's enrolment. Numbers and strings."""
    out: dict = {"model": model, "dim": fg.model_dim(model),
                 "published_cosine_same": fg.cosine_same(model),
                 "loaded": False, "generation": 0, "labels": [],
                 "samples": 0, "generations_on_disk": [],
                 "foreign_generations": 0, "foreign_samples": 0,
                 "reason": ""}
    gallery = fg.FaceGallery(root=root, model=model)
    out["generations_on_disk"] = [int(g) for g in gallery.generations()]
    if not gallery.load():
        out["foreign_generations"] = len(gallery.foreign_generations)
        out["foreign_samples"] = gallery.foreign_sample_count()
        out["reason"] = (gallery.reenrol_message()
                         if gallery.foreign_generations
                         else "nothing enrolled for %s" % model)
        return out
    out["loaded"] = True
    out["generation"] = int(gallery.loaded_generation)
    out["labels"] = list(gallery.labels())
    out["samples"] = int(gallery.total())
    out["reason_recorded"] = str(gallery.provenance().get("reason", ""))

    pools, poses, notes = {}, {}, 0
    for label in gallery.labels():
        vecs = gallery.embeddings(label)
        takes = gallery.takes(label)
        pools[label] = vecs
        poses[label] = [pose_of(t, profile_deg) for t in takes]
        notes += sum(1 for t in takes if t.note)
    out["notes_recorded"] = int(notes)
    out["angles_recorded"] = int(sum(1 for ps in poses.values()
                                     for p in ps if p != "unrecorded"))

    # ---- intra-person, overall and by pose
    per_label = {}
    all_intra: list = []
    frontal_pairs: list = []
    mixed_pairs: list = []
    profile_pairs: list = []
    for label, vecs in pools.items():
        cs = cosines(vecs)
        all_intra.extend(cs)
        per_label[label] = {"takes": len(vecs), "pairs": len(cs),
                            "cosine": spread(cs),
                            "frontal_takes": poses[label].count("frontal"),
                            "profile_takes": poses[label].count("profile"),
                            "unrecorded_takes": poses[label].count(
                                "unrecorded")}
        for i, j in itertools.combinations(range(len(vecs)), 2):
            pi, pj = poses[label][i], poses[label][j]
            if "unrecorded" in (pi, pj):
                continue
            value = fg.cosine(vecs[i], vecs[j])
            if pi == pj == "frontal":
                frontal_pairs.append(value)
            elif pi == pj == "profile":
                profile_pairs.append(value)
            else:
                mixed_pairs.append(value)
    out["per_label"] = per_label
    out["intra"] = spread(all_intra)
    out["intra_frontal_frontal"] = spread(frontal_pairs)
    out["intra_frontal_profile"] = spread(mixed_pairs)
    out["intra_profile_profile"] = spread(profile_pairs)

    # ---- every non-matching pair the gallery itself can offer
    cross: list = []
    labels = list(pools)
    for a, b in itertools.combinations(labels, 2):
        cross.extend(cosines(pools[a], pools[b]))
    out["cross_person"] = spread(cross)
    out["cross_person_labels"] = len(labels)
    return out


# ------------------------------------------- the only "not him" we can make
def synthetic_families(seed: int = 7, size: int = 112) -> dict:
    """Four families of non-face crops, generated here by numpy.

    These are the only NON-MATCHING inputs available when one person is
    enrolled, and they answer a real question: what does this model score on
    something that is not a face at all? The answer bounds the false-positive
    side and nothing else -- a drawn shape is not a stranger's face, and this
    script will not pretend otherwise.
    """
    rng = np.random.default_rng(seed)
    out = {}
    out["uniform noise"] = [rng.integers(0, 256, (size, size, 3), np.uint8)
                            for _ in range(OOD_N)]
    out["flat colour"] = [np.full((size, size, 3),
                                  rng.integers(0, 256, 3, dtype=np.uint8),
                                  np.uint8) for _ in range(OOD_N)]
    blobs = []
    yy, xx = np.mgrid[:size, :size]
    for _ in range(OOD_N):
        img = np.zeros((size, size, 3), np.uint8)
        for _b in range(int(rng.integers(3, 8))):
            cx, cy = rng.integers(0, size, 2)
            r = int(rng.integers(6, 30))
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.integers(
                0, 256, 3)
        blobs.append(img)
    out["random blobs"] = blobs
    grads = []
    ramp = np.linspace(0, 1, size)[None, :, None]
    for _ in range(OOD_N):
        a, b = rng.random(3) * 255.0, rng.random(3) * 255.0
        img = a[None, None, :] * (1 - ramp) + b[None, None, :] * ramp
        grads.append(np.repeat(img, size, axis=0).astype(np.uint8))
    out["smooth gradients"] = grads
    return out


def arcface_embedder(model_dir):
    """``(embed, ms_per_call, reason)`` for ArcFace, or (None, 0.0, why)."""
    ok, why = fm.verify(fm.ARCFACE_MBF, model_dir=model_dir)
    if not ok:
        return None, 0.0, why
    try:
        import onnxruntime as ort                # noqa: PLC0415
    except Exception as exc:                     # noqa: BLE001
        return None, 0.0, "onnxruntime: %s" % exc
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(fm.model_path(fm.ARCFACE_MBF, model_dir)),
                                opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def embed(crop):
        raw = sess.run(None, {name: fi.arcface_blob(crop)})[0]
        return fi.normalise(np.asarray(raw).ravel())

    return embed, 0.0, ""


def sface_embedder(model_dir):
    """``(embed, 0.0, reason)`` for SFace, or (None, 0.0, why)."""
    ok, why = fm.verify(fm.SFACE, model_dir=model_dir)
    if not ok:
        return None, 0.0, why
    try:
        import cv2                               # noqa: PLC0415
    except Exception as exc:                     # noqa: BLE001
        return None, 0.0, "cv2: %s" % exc
    try:
        net = cv2.FaceRecognizerSF_create(
            str(fm.model_path(fm.SFACE, model_dir)), "")
    except Exception as exc:                     # noqa: BLE001
        return None, 0.0, "cv2 could not build SFace: %s" % exc

    def embed(crop):
        return np.asarray(net.feature(crop), dtype=np.float32).ravel()

    return embed, 0.0, ""


def ood_probe(embed, families: dict) -> dict:
    """Pairwise cosine between UNRELATED non-face crops, per family, plus ms.

    A high number here is the model saying two unrelated pieces of nothing
    are the same person. SFace does that (measured 2026-09-02); ArcFace does
    it MORE (measured 2026-09-03). It is why identity is only ever computed
    from a detection that already cleared the confidence bar, and it is worth
    re-measuring on his box because it is the false-positive floor every
    identity threshold sits above.
    """
    out: dict = {}
    times: list = []
    for family, images in families.items():
        vecs = []
        for img in images:
            t0 = time.perf_counter()
            vecs.append(embed(img))
            times.append((time.perf_counter() - t0) * 1000.0)
        out[family] = spread(cosines(vecs))
    out["_embed_ms"] = spread(times[len(families):] or times)
    return out


# ---------------------------------------------------------------- the bar
def suggest_bar(model: dict) -> dict:
    """A starting identity bar from FACE evidence only, with its counts.

    THE SYNTHETIC CROPS ARE DELIBERATELY NOT IN THIS CALCULATION, and getting
    that wrong makes the whole instrument useless. Both models score unrelated
    non-face crops near 1.0 against each other, so folding them in makes every
    verdict read "no separation" even when two real people separate perfectly
    (measured on a simulated gallery: another person at -0.05 against a
    same-person floor of 0.65, while the synthetic ceiling was 0.98).

    They are two different guards answering two different questions. The
    COSINE BAR asks "which enrolled person is this face"; the CONFIDENCE AND
    LANDMARK GATES ask "is this a face at all". No cosine bar can do the
    second job -- that is the whole finding -- so the probe is reported on its
    own, underneath, as the reason those gates may never be relaxed.

    NOT A MEASUREMENT OF ACCURACY either way. It is the midpoint between the
    worst same-person pair and the best different-person pair, and it is only
    as good as the counts printed beside it.
    """
    out = {"floor": 0.0, "ceiling": 0.0, "suggested": 0.0, "basis": "",
           "separation": 0.0, "ok": False}
    intra = model.get("intra") or {}
    if not intra.get("n"):
        out["basis"] = "no same-person pairs: fewer than two takes enrolled"
        return out
    out["floor"] = float(intra["min"])
    cross = model.get("cross_person") or {}
    if not cross.get("n"):
        out["basis"] = ("only one person is enrolled, so there is no "
                        "different-person evidence to set a ceiling from")
        return out
    out["ceiling"] = float(cross["max"])
    out["basis"] = "another enrolled person"
    out["separation"] = out["floor"] - out["ceiling"]
    out["suggested"] = (out["floor"] + out["ceiling"]) / 2.0
    out["ok"] = bool(out["separation"] > 0.0)
    return out


def ood_verdict(ood: dict) -> dict:
    """What the model does on inputs that are not faces at all.

    ``worst`` is the highest cosine between two UNRELATED pieces of nothing.
    A number near 1 means the model would confidently call them the same
    person, which is why nothing may compute an embedding from a detection
    that has not already cleared the confidence bar and the landmark test.
    """
    out = {"worst": 0.0, "family": "", "n": 0, "ok": False}
    worst = None
    for family, s in ood.items():
        if family.startswith("_") or not s["n"]:
            continue
        out["n"] += int(s["n"])
        if worst is None or s["max"] > worst[1]:
            worst = (family, float(s["max"]))
    if worst is None:
        return out
    out["family"], out["worst"] = worst
    out["ok"] = True
    return out


# ------------------------------------------------------------ the printing
def line_for(name: str, s: dict) -> str:
    if not s["n"]:
        return "  %-26s        --  (no pairs)" % name
    return ("  %-26s n=%4d  min %6.3f  p05 %6.3f  p50 %6.3f  max %6.3f"
            % (name, s["n"], s["min"], s["p05"], s["p50"], s["max"]))


def report(payload: dict, say) -> None:
    say("face model comparison -- numbers only. No frame is opened, decoded, "
        "displayed or written; every")
    say("line below is a count, a cosine, a degree or a millisecond.")
    say("")
    say("gallery      %s" % payload["gallery"])
    say("backend now  %s   (camera.face_backend=%r)"
        % (payload["backend"], payload["configured_backend"]))
    say("profile cut  |yaw| > %.1f deg" % payload["profile_deg"])
    say("")
    for model in payload["models"]:
        say("=" * 72)
        say("%s   %d-D   published same-person cosine: %s"
            % (model["model"].upper(), model["dim"],
               ("%.3f" % model["published_cosine_same"])
               if model["published_cosine_same"] is not None
               else "NONE -- never measured for this checkpoint"))
        if not model["loaded"]:
            say("  NOT ENROLLED. %s" % model["reason"])
            if model.get("ood"):
                say("")
                say("  what it scores on things that are NOT faces "
                    "(synthetic, generated here):")
                for family, s in sorted(model["ood"].items()):
                    if not family.startswith("_"):
                        say(line_for(family, s))
                say("  %-26s p50 %6.2f ms"
                    % ("cost per embedding", model["ood"]["_embed_ms"]["p50"]))
            say("")
            continue
        say("  generation %d, %d take(s) over %d label(s): %s"
            % (model["generation"], model["samples"], len(model["labels"]),
               ", ".join(model["labels"])))
        say("  %d take(s) carry a head angle, %d carry a note"
            % (model["angles_recorded"], model["notes_recorded"]))
        say("")
        say("  SAME PERSON -- the floor the identity bar has to sit under")
        say(line_for("all pairs", model["intra"]))
        if model["angles_recorded"] == 0:
            say("  THE SIDE-ANGLE QUESTION CANNOT BE ANSWERED FROM THIS "
                "GALLERY.")
            say("  Not one take records the head angle it was captured at, "
                "so there is no")
            say("  way to separate the profile takes from the frontal ones. "
                "Re-enrolling")
            say("  records the angle on every take (jarvis/faceenrol.py), "
                "and then this")
            say("  section answers it.")
        else:
            say(line_for("frontal vs frontal", model["intra_frontal_frontal"]))
            say(line_for("frontal vs profile", model["intra_frontal_profile"]))
            say(line_for("profile vs profile", model["intra_profile_profile"]))
        say("")
        say("  NOT HIM -- every non-matching thing available")
        if model["cross_person_labels"] < 2:
            say("  %-26s        --  (only one person is enrolled)"
                % "another person")
        else:
            say(line_for("another enrolled person", model["cross_person"]))
        for family, s in sorted((model.get("ood") or {}).items()):
            if not family.startswith("_"):
                say(line_for("synthetic: " + family, s))
        if model.get("ood"):
            say("  %-26s p50 %6.2f ms"
                % ("cost per embedding", model["ood"]["_embed_ms"]["p50"]))
        say("")
        bar = model["bar"]
        if not bar["ok"] and not bar["ceiling"]:
            say("  BAR       CANNOT BE SET FROM THIS GALLERY -- %s"
                % bar["basis"])
            if bar["floor"]:
                say("            worst same-person pair %.3f, over %d pair(s)."
                    % (bar["floor"], model["intra"]["n"]))
        elif not bar["ok"]:
            say("  BAR       THE TWO OVERLAP. worst same-person %.3f is BELOW "
                "best other-person %.3f;" % (bar["floor"], bar["ceiling"]))
            say("            no single cosine separates them, so some take is "
                "hurting more than helping.")
        else:
            say("  BAR       worst same-person %.3f, best other-person %.3f "
                "(%s)" % (bar["floor"], bar["ceiling"], bar["basis"]))
            say("            separation %.3f -- a starting "
                "camera.identity_min of %.3f"
                % (bar["separation"], bar["suggested"]))
            say("            A STARTING POINT resting on %d same-person and "
                "%d other-person pair(s)."
                % (model["intra"]["n"], model["cross_person"]["n"]))
            say("            It is not an accuracy measurement and it is not "
                "a promise.")
        ver = model.get("ood_verdict") or {}
        if ver.get("ok"):
            say("  NOT A FACE  two unrelated synthetic crops score up to "
                "%.3f against each other" % ver["worst"])
            say("            (%s, %d pairs). NO COSINE BAR CAN FIX THAT. It "
                "is why an" % (ver["family"], ver["n"]))
            say("            embedding is only ever taken from a detection "
                "that already cleared")
            say("            camera.min_conf and the landmark test -- those "
                "gates, not this bar.")
        say("")
    say("=" * 72)
    for note in payload["notes"]:
        say("* %s" % note)


# ------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gallery", default="",
                    help="gallery directory (default: the configured one)")
    ap.add_argument("--profile-deg", type=float, default=PROFILE_DEG,
                    help="beyond this |yaw| a take counts as profile "
                         "(default %.0f)" % PROFILE_DEG)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the synthetic out-of-distribution probe, "
                         "which needs the weights")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable payload on stdout and nothing else")
    return ap


def collect(args, cfg) -> dict:
    root = Path(args.gallery) if args.gallery else Path(PATHS.FACE_GALLERY)
    model_dir = str(cfg.get("camera.model_dir", "") or "") or None
    configured = str(cfg.get("camera.face_backend", "") or "")
    payload: dict = {
        "gallery": str(root),
        "profile_deg": float(args.profile_deg),
        "configured_backend": configured,
        "backend": fm.backend_for(configured).name,
        "models": [], "notes": [],
    }
    families = None if args.no_probe else synthetic_families()
    makers = {fg.SFACE_MODEL: sface_embedder,
              fg.ARCFACE_MODEL: arcface_embedder}
    for model in (fg.SFACE_MODEL, fg.ARCFACE_MODEL):
        entry = one_model(root, model, float(args.profile_deg))
        ood: dict = {}
        if families is not None:
            embed, _ms, why = makers[model](model_dir)
            if embed is None:
                entry["probe_reason"] = why
            else:
                ood = ood_probe(embed, families)
        entry["ood"] = ood
        entry["ood_verdict"] = ood_verdict(ood)
        entry["bar"] = suggest_bar(entry)
        payload["models"].append(entry)

    payload["notes"].append(
        "The two models are never compared with each other. An SFace "
        "128-vector and an ArcFace 512-vector are not weakly comparable; the "
        "cosine between them measures nothing.")
    payload["notes"].append(
        "A high synthetic score is the model calling two pieces of nothing "
        "the same person. It is why identity is only ever computed from a "
        "detection that already cleared camera.min_conf.")
    if any(m["loaded"] and m["angles_recorded"] == 0
           for m in payload["models"]):
        payload["notes"].append(
            "At least one enrolment records no head angles, so the "
            "side-angle question is unanswered. Re-enrol to record them.")
    if not any(m["loaded"] for m in payload["models"]):
        payload["notes"].append(
            "Nothing is enrolled for either model, so there are no "
            "same-person numbers at all. Enrol first.")
    return payload


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    hits = self_check()
    if hits:
        print("REFUSING TO RUN: this file's code names a pixel-bearing "
              "call: %s" % ", ".join(hits))
        return 4
    cfg = AssistantConfig.load()
    payload = collect(args, cfg)
    assert_numbers_only(payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        report(payload, print)
    return 0 if any(m["loaded"] for m in payload["models"]) else 1


if __name__ == "__main__":
    sys.exit(main())
