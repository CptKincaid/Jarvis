"""Enrolling his face: the numbers that decide whether a gallery is worth
saving, and the guided capture that produces them.

THE WORKFLOW THIS MODULE IS SHAPED BY. Hunter's standing rule is that nothing
may look at what his camera sees. So enrolment cannot be "watch the preview
until it looks right" -- it has to be a run whose ENTIRE output is numbers he
can paste to somebody who then judges the enrolment without ever seeing him.
Everything below exists to make that one workflow work: per sample a
confidence, a face size, an interocular distance, a head angle, a sharpness
and an accept/reject with a reason; per gallery a sample count, a pose spread
and a pairwise-cosine distribution; and a verdict on whether the result is
good enough to use. ``jarvis/visionrig.assert_numbers_only`` is run over the
report, so this is enforced rather than promised.

NO CROP AND NO FRAME REACHES THE DISK. The only artefact written is the
128-float embedding, into ``jarvis/facegallery.py``. Crops exist inside
``SFaceRecogniser.embed`` and inside ``sharpness()`` and die with the call;
nothing here holds a frame, a crop or an embedding after ``offer()`` returns.

WHY IDENTITY IS ONLY EVER COMPUTED FROM A DETECTION THAT CLEARED THE BAR.
SFace's 128-D embedding collapses on out-of-distribution input: measured on
this box 2026-09-02, unrelated NON-FACE crops match each other at mean cosine
0.66-0.92 with 94-100% of pairs above the 0.363 "same person" bar
(jarvis/facemodels.py). A bad crop, a motion blur or a false-positive box
therefore does not score LOW against a gallery -- it scores CONFIDENTLY. So
the detector is the only thing standing between a bad crop and a poisoned
gallery, and a sample that failed ANY quality gate never has an embedding
computed at all. That is why ``EnrolmentSession.offer`` judges first and
embeds second, and why the recogniser re-checks the same bar underneath it
(jarvis/facedetect.py:212-226). Two gates on one rule is deliberate: this is
not a rule to leave to a caller.

HOW MANY SAMPLES, AND WHY THAT MANY. His voiceprint is the precedent that
exists: 11 ECAPA embeddings with pairwise cosine 0.289-0.735 -- a real
SPREAD, not eleven readings of one sentence, and the enrolment script that
produced it varies distance, pace and loudness on purpose
(scripts/enroll_voice.py:41-61). The face equivalent of "one sentence eleven
times" is one head position, and it is the failure mode his own usage
guarantees: measured on his camera 2026-09-02, his yaw is ~14 deg when he
looks at the lens and ~54 deg when he looks at his screen. A gallery built at
14 deg is a gallery that stops working the moment he turns to work. So the
default plan is 13 samples over five stations that walk the yaw range he
actually occupies, and ``judge_gallery`` refuses to save a pool that did not
achieve the spread -- an enrolment that merely FINISHED is not an enrolment
that WORKS.

WHAT THE TWO REFUSALS MEAN.

* **Too tight** -- the samples are near-duplicates (pairwise cosine p50 above
  ``NEAR_DUPLICATE_MAX``, or a yaw range under ``MIN_YAW_SPREAD_DEG``). This
  gallery will match him in one pose and reject him in every other, which
  reads at the far end as "Jarvis does not know me any more" and is the
  expensive failure: it is silent, and it looks like the camera not working.
* **Too loose** -- some sample does not cluster with the rest (its median
  cosine to the others is under the "same person" bar). That is either a
  different person who walked past, or a crop that is not a face. Either way
  it drags every future match toward itself, because ``FaceGallery.match``
  scores against the pool's BEST member.

Nothing here imports cv2, torch or a model. The detector, the recogniser and
the frame source are all seams, so the whole module runs in the suite with no
camera, no weights, no display and no GPU.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from jarvis.facegallery import SFACE_COSINE_SAME, FaceGallery, cosine
from jarvis.logs import get_logger
from jarvis.visionrig import IDX_SCORE, Check, HeadModel, _pct, observe

log = get_logger("faceenrol")

# ---------------------------------------------------------------- the bars
# SFace's input is 112x112 and ``alignCrop`` warps the detected face into it.
# A capture-space face smaller than that is being UPSAMPLED into the model,
# which invents detail the lens never resolved. docs/vision.md section 9 calls
# a 53 px face "far too small" for exactly this reason, and the model costs
# the same either way, so a small take is pure loss.
MIN_FACE_PX = 112.0
# Interocular distance, in capture pixels. The 5-point template SFace aligns
# to puts the eye centres 35.2 px apart in the 112x112 crop (ArcFace's
# canonical destination points, x = 38.29 and 73.53). Under that the aligner
# is magnifying, which is the same loss as above measured where it actually
# bites. His measured interocular is 140 px facing the lens, so this is a long
# way below anything his own mount produces -- it is here to catch a face at
# the far end of the room, not to second-guess his desk.
MIN_EYE_PX = 35.0
# Beyond this the far eye starts to occlude, the interocular foreshortens past
# the point where the landmark ratio means anything, and SFace has less of a
# face to work with. His working pose is ~54 deg, so the bar has to sit above
# it or enrolment refuses his most common head position.
MAX_YAW_DEG = 62.0
# A head tilted this far at a desk is a head resting on a hand, and the crop
# SFace aligns from is rotated by the same amount.
MAX_ROLL_DEG = 30.0
# Gradient energy over the face box, normalised by its own mean level (see
# ``sharpness``). PROVISIONAL AND DELIBERATELY LOW: nobody has measured this
# on his face, and the asymmetry is one-sided -- a bar set too high makes
# enrolment impossible with a confusing message, while a bar set too low lets
# a slightly soft sample through where the cohesion check catches it. So it
# catches gross motion blur and nothing finer, the report prints the whole
# distribution, and --min-sharpness moves it once there is data.
MIN_SHARPNESS = 0.015

# 13 samples over the default plan; 8 is the floor docs/vision.md section 5
# states ("8-12 takes"), and the voiceprint precedent is 11.
MIN_SAMPLES = 8
DEFAULT_SAMPLES = 13
# Pairwise cosine p50 above this and the pool is one face repeated: the
# samples carry no more information than the first of them.
NEAR_DUPLICATE_MAX = 0.98
# His usage spans ~14 deg (at the lens) to ~54 deg (at his screen), so a
# gallery that does not cover at least 25 deg of that has not covered the
# thing that varies.
MIN_YAW_SPREAD_DEG = 25.0
FRONTAL_MAX_DEG = 15.0     # "looking at the camera"
OFFAXIS_MIN_DEG = 20.0     # "looking at the screen"
MIN_FRONTAL = 2
MIN_OFFAXIS = 2
# Two faces in frame during enrolment is the one thing that must never be
# quietly averaged in: a gallery that learned a visitor identifies the wrong
# person confidently, and nothing about it looks wrong afterwards.
MAX_FACES_IN_FRAME = 1


@dataclass(frozen=True)
class SampleLimits:
    """The per-sample bars. ``min_conf`` has NO default: it is the detector
    confidence bar from his config, and it is the only thing standing between
    a bad crop and a poisoned gallery (see the module docstring)."""

    min_conf: float
    min_face_px: float = MIN_FACE_PX
    min_eye_px: float = MIN_EYE_PX
    max_yaw_deg: float = MAX_YAW_DEG
    max_roll_deg: float = MAX_ROLL_DEG
    min_sharpness: float = MIN_SHARPNESS
    max_faces: int = MAX_FACES_IN_FRAME


@dataclass(frozen=True)
class Station:
    """One instruction and the head angles it is meant to produce.

    A station GUIDES; it does not judge. ``judge_gallery`` judges the pose
    distribution that actually came out, because a station whose window his
    desk cannot produce -- the camera clamped on the side his screen is not
    on -- must cost that station and not the enrolment.
    """

    key: str
    prompt: str
    yaw_lo: float          # signed, inclusive
    yaw_hi: float
    samples: int

    def wants(self, yaw_deg: float) -> bool:
        return self.yaw_lo <= float(yaw_deg) <= self.yaw_hi


# Five stations, 13 samples. The two that matter most are `lens` and `screen`
# -- the two poses he measurably occupies. `across` is the other side of the
# screen and is the one most likely to be unreachable on a given mount, which
# is why nothing downstream requires it by name. `back` and `close` vary
# DISTANCE rather than angle: face_px is reported per sample and its range is
# printed, so a plan that produced no distance variation is visible even
# though nothing refuses it.
DEFAULT_PLAN: Tuple[Station, ...] = (
    Station("lens", "Look straight into the camera lens.", -15.0, 15.0, 3),
    Station("screen",
            "Now look at your screen exactly as you do when you are working.",
            18.0, 62.0, 3),
    Station("across",
            "Turn your head the OTHER way -- as if looking at the far side "
            "of the desk.", -62.0, -18.0, 3),
    Station("back",
            "Back to the camera, but sit back -- further away than usual.",
            -15.0, 15.0, 2),
    Station("close",
            "Lean in closer than usual, still looking at the camera.",
            -15.0, 15.0, 2),
)


@dataclass(frozen=True)
class Sample:
    """One candidate frame, as numbers. There is deliberately no crop, no
    landmark array and no embedding on this object: what it holds is what may
    be printed, and that is the whole design (the same rule
    ``visionrig.FaceObservation`` is written to)."""

    index: int
    station: str
    faces: int
    conf: float
    face_px: float
    eye_px: float
    yaw_deg: float
    roll_deg: float
    bearing_deg: float
    sharpness: float
    accepted: bool
    reason: str

    def as_dict(self) -> dict:
        return {"index": self.index, "station": self.station,
                "faces": self.faces, "conf": self.conf,
                "face_px": self.face_px, "eye_px": self.eye_px,
                "yaw_deg": self.yaw_deg, "roll_deg": self.roll_deg,
                "bearing_deg": self.bearing_deg,
                "sharpness": self.sharpness, "accepted": self.accepted,
                "reason": self.reason}

    def line(self) -> str:
        return ("  %2d %-7s %s conf %.2f  face %4.0fpx  eyes %3.0fpx  "
                "yaw %+5.1f  roll %+5.1f  sharp %.3f  %s"
                % (self.index, self.station,
                   "KEEP" if self.accepted else "drop", self.conf,
                   self.face_px, self.eye_px, self.yaw_deg, self.roll_deg,
                   self.sharpness, self.reason))


# ------------------------------------------------------------- the measures
def sharpness(frame, x: float, y: float, w: float, h: float) -> float:
    """Gradient energy over the face box, normalised by its mean level.

    ONE FLOAT COMES OUT AND THE CROP DIES HERE. That is the whole contract:
    a blur measure needs pixels, so the pixels are read inside this function,
    reduced to a scalar, and never returned, stored or logged.

    Normalising by the mean makes it independent of exposure, and the box is
    subsampled to roughly 64 px on its long side first so that a 300 px face
    and a 130 px face give comparable numbers -- otherwise the measure falls
    with distance and a "blur" bar would really be a distance bar. Gross
    motion blur is what survives that normalisation, and gross motion blur is
    all this claims to catch.
    """
    try:
        arr = np.asarray(frame)
        if arr.ndim < 2:
            return 0.0
        ih, iw = int(arr.shape[0]), int(arr.shape[1])
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(iw, int(x + w)), min(ih, int(y + h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return 0.0
        box = arr[y0:y1, x0:x1]
        step = max(1, min(x1 - x0, y1 - y0) // 64)
        box = box[::step, ::step]
        box = np.asarray(box, dtype=np.float32)
        if box.ndim == 3:
            box = box.mean(axis=2)
        if box.shape[0] < 4 or box.shape[1] < 4:
            return 0.0
        level = float(box.mean())
        if level <= 1e-6:
            return 0.0
        gx = float(np.abs(np.diff(box, axis=1)).mean())
        gy = float(np.abs(np.diff(box, axis=0)).mean())
        return (gx + gy) / (2.0 * level)
    except Exception:  # noqa: BLE001 - a weird frame is not a crash
        log.debug("faceenrol: sharpness failed on a frame", exc_info=True)
        return 0.0


def judge_sample(obs, faces: int, sharp: float,
                 limits: SampleLimits) -> Tuple[bool, str]:
    """``(accepted, reason)`` for one observation. The reason is always set,
    including on acceptance, because a report whose good lines are blank
    teaches nobody what "good" looked like.

    THE CONFIDENCE BAR IS CHECKED FIRST AND IS NOT NEGOTIABLE. Everything
    below it is a quality preference; that one is the rule that keeps SFace's
    out-of-distribution collapse out of the gallery.
    """
    if faces > int(limits.max_faces):
        return False, ("%d faces in frame -- a gallery that learns a visitor "
                       "identifies the wrong person confidently" % faces)
    if obs.conf < limits.min_conf:
        return False, ("conf %.2f under the %.2f detector bar -- no "
                       "embedding is computed from this"
                       % (obs.conf, limits.min_conf))
    if not obs.landmarks_ok:
        return False, "the eye landmarks coincide; the crop cannot be aligned"
    if obs.face_px < limits.min_face_px:
        return False, ("face %.0f px under SFace's %.0f px input -- the crop "
                       "would be upsampled" % (obs.face_px,
                                               limits.min_face_px))
    if obs.eye_px < limits.min_eye_px:
        return False, ("interocular %.0f px under %.0f -- too small to align"
                       % (obs.eye_px, limits.min_eye_px))
    if abs(obs.yaw_deg) > limits.max_yaw_deg:
        return False, ("yaw %+.1f deg past the %.0f deg limit"
                       % (obs.yaw_deg, limits.max_yaw_deg))
    if abs(obs.roll_deg) > limits.max_roll_deg:
        return False, ("roll %+.1f deg past the %.0f deg limit"
                       % (obs.roll_deg, limits.max_roll_deg))
    if sharp < limits.min_sharpness:
        return False, ("sharpness %.3f under %.3f -- motion blur"
                       % (sharp, limits.min_sharpness))
    return True, "ok"


# ----------------------------------------------------------- the statistics
def pairwise_cosines(embs: Sequence) -> List[float]:
    """Every unordered pair, once. n=13 gives 78 numbers, which is the
    distribution the whole verdict is read off."""
    out: List[float] = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            out.append(cosine(embs[i], embs[j]))
    return out


def cohesion(embs: Sequence) -> List[float]:
    """Per sample, its MEDIAN cosine to the others.

    Median rather than mean: one genuinely distant pose should not condemn a
    sample, but a sample that is unlike MOST of the pool is either a
    different person or a crop that is not a face. And per sample rather than
    over the pool, because the pool's own median stays healthy while one
    poisoned member sits in it -- and ``FaceGallery.match`` scores against the
    pool's BEST member, so one poisoned member is all it takes.
    """
    out: List[float] = []
    for i, emb in enumerate(embs):
        others = [cosine(emb, e) for j, e in enumerate(embs) if j != i]
        out.append(_pct(others, 0.5) if others else 0.0)
    return out


def judge_gallery(samples: Sequence[Sample], embs: Sequence,
                  identity_min: float = SFACE_COSINE_SAME,
                  min_samples: int = MIN_SAMPLES) -> Tuple[Check, ...]:
    """The pass/fail lines that decide whether this gallery may be saved.

    A gallery that is too tight fails him in every pose but one; a gallery
    that is too loose has somebody else in it. Both are worse than no gallery,
    because both are SILENT -- and the second is worse than the first,
    because it is confident.
    """
    kept = [s for s in samples if s.accepted]
    checks: List[Check] = [
        Check("samples", len(embs) >= int(min_samples),
              "%d embeddings from %d candidate frames, against a floor of %d"
              % (len(embs), len(samples), int(min_samples)))]
    if not embs:
        checks.append(Check("pose_spread", None, "no samples to measure"))
        checks.append(Check("variation", None, "no samples to compare"))
        checks.append(Check("cohesion", None, "no samples to compare"))
        return tuple(checks)

    yaws = [s.yaw_deg for s in kept]
    spread = (max(yaws) - min(yaws)) if yaws else 0.0
    frontal = sum(1 for y in yaws if abs(y) <= FRONTAL_MAX_DEG)
    offaxis = sum(1 for y in yaws if abs(y) >= OFFAXIS_MIN_DEG)
    checks.append(Check(
        "pose_spread",
        (spread >= MIN_YAW_SPREAD_DEG and frontal >= MIN_FRONTAL
         and offaxis >= MIN_OFFAXIS),
        "yaw %+.1f..%+.1f deg (%.1f deg of spread, want %.0f), %d frontal "
        "(<=%.0f deg, want %d), %d off-axis (>=%.0f deg, want %d). He is at "
        "~14 deg looking at the lens and ~54 at his screen, so a gallery "
        "without both fails the moment he turns to work."
        % (min(yaws) if yaws else 0.0, max(yaws) if yaws else 0.0, spread,
           MIN_YAW_SPREAD_DEG, frontal, FRONTAL_MAX_DEG, MIN_FRONTAL,
           offaxis, OFFAXIS_MIN_DEG, MIN_OFFAXIS)))

    pairs = pairwise_cosines(embs)
    p50 = _pct(pairs, 0.5)
    checks.append(Check(
        "variation", (not pairs) or p50 <= NEAR_DUPLICATE_MAX,
        "%d pairs, cosine min %.3f p50 %.3f max %.3f. Above %.2f at the "
        "median the samples are one face repeated and carry no more "
        "information than the first of them."
        % (len(pairs), min(pairs) if pairs else 0.0, p50,
           max(pairs) if pairs else 0.0, NEAR_DUPLICATE_MAX)))

    coh = cohesion(embs)
    worst = min(range(len(coh)), key=lambda i: coh[i]) if coh else 0
    # The embeddings are in accepted order; the SAMPLES are numbered over
    # every candidate including the dropped ones. Printing the embedding's
    # index would name a different line of the report as soon as one frame
    # was rejected, which for a report whose whole job is to be pasted and
    # read by somebody else is worse than useless.
    named = kept[worst].index if worst < len(kept) else worst
    checks.append(Check(
        "cohesion", (len(coh) < 2) or min(coh) >= float(identity_min),
        "worst sample is #%d at median cosine %.3f to the others, against "
        "OpenCV's %.3f 'same person' bar for SFace; the pool's own median is "
        "%.3f. A sample under the bar is a different person or a crop that "
        "is not a face, and match() scores against the pool's BEST member."
        % (named, min(coh) if coh else 0.0, float(identity_min),
           _pct(coh, 0.5))))
    return tuple(checks)


# --------------------------------------------------------------- the report
@dataclass
class EnrolmentReport:
    """Everything the enrolment says, and it says nothing else.

    Every field is a count, an angle, a size, a score, a millisecond or a
    verdict. ``jarvis/visionrig.assert_numbers_only`` is run over
    ``to_dict()`` before anything is printed, so "he can paste this to
    anybody" is a property of the structure rather than of the print
    statements.
    """

    label: str = ""
    frames: int = 0
    offered: int = 0
    accepted: int = 0
    rejected: int = 0
    stations_run: int = 0
    stations_planned: int = 0
    seconds: float = 0.0
    detector_ok: bool = False
    reason: str = ""
    min_conf: float = 0.0
    identity_min: float = 0.0
    conf_min: float = 0.0
    conf_p50: float = 0.0
    face_px_min: float = 0.0
    face_px_p50: float = 0.0
    face_px_max: float = 0.0
    eye_px_min: float = 0.0
    eye_px_p50: float = 0.0
    yaw_min: float = 0.0
    yaw_p50: float = 0.0
    yaw_max: float = 0.0
    yaw_spread: float = 0.0
    frontal: int = 0
    offaxis: int = 0
    roll_min: float = 0.0
    roll_max: float = 0.0
    sharp_min: float = 0.0
    sharp_p50: float = 0.0
    embed_ms_p50: float = 0.0
    pairs: int = 0
    cos_min: float = 0.0
    cos_p05: float = 0.0
    cos_p50: float = 0.0
    cos_max: float = 0.0
    cohesion_min: float = 0.0
    cohesion_p50: float = 0.0
    rejects: tuple = ()          # (reason-word, count) pairs
    samples: tuple = ()
    checks: tuple = ()
    saved_generation: int = 0
    gallery_total: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.detector_ok
                    and all(c.ok is not False for c in self.checks))

    def to_dict(self) -> dict:
        out = {k: v for k, v in vars(self).items()
               if k not in ("checks", "samples", "rejects")}
        out["ok"] = self.ok
        out["checks"] = [list(c.as_tuple()) for c in self.checks]
        out["samples"] = [s.as_dict() for s in self.samples]
        out["rejects"] = [[str(k), int(v)] for k, v in self.rejects]
        return out

    def lines(self) -> list:
        out = [
            "label      %s" % (self.label or "-"),
            "detector   ok=%s%s" % (self.detector_ok,
                                    ("  (%s)" % self.reason)
                                    if self.reason else ""),
            "frames     %d read, %d offered, %d kept, %d dropped, %.1fs"
            % (self.frames, self.offered, self.accepted, self.rejected,
               self.seconds),
            "stations   %d of %d run" % (self.stations_run,
                                         self.stations_planned),
            "confidence min %.3f  p50 %.3f  against a %.2f bar"
            % (self.conf_min, self.conf_p50, self.min_conf),
            "face size  %.0f..%.0f px (p50 %.0f)  interocular %.0f px "
            "(p50 %.0f)" % (self.face_px_min, self.face_px_max,
                            self.face_px_p50, self.eye_px_min,
                            self.eye_px_p50),
            "yaw        %+.1f..%+.1f deg (spread %.1f)  %d frontal  "
            "%d off-axis" % (self.yaw_min, self.yaw_max, self.yaw_spread,
                             self.frontal, self.offaxis),
            "roll       %+.1f..%+.1f deg" % (self.roll_min, self.roll_max),
            "sharpness  min %.3f  p50 %.3f" % (self.sharp_min,
                                               self.sharp_p50),
            "embedding  p50 %.1f ms" % self.embed_ms_p50,
            "pairwise   %d pairs  cosine min %.3f p05 %.3f p50 %.3f max %.3f"
            % (self.pairs, self.cos_min, self.cos_p05, self.cos_p50,
               self.cos_max),
            "cohesion   worst %.3f  p50 %.3f  against the %.3f same-person "
            "bar" % (self.cohesion_min, self.cohesion_p50,
                     self.identity_min),
        ]
        if self.rejects:
            out.append("dropped    " + ", ".join(
                "%s x%d" % (k, v) for k, v in self.rejects))
        out.append("samples")
        out.extend(s.line() for s in self.samples)
        for check in self.checks:
            mark = "n/a " if check.ok is None else ("PASS" if check.ok
                                                    else "FAIL")
            out.append("  [%s] %-13s %s" % (mark, check.name, check.detail))
        if self.saved_generation:
            out.append("SAVED      generation %d, %d embeddings in the gallery"
                       % (self.saved_generation, self.gallery_total))
        out.append("VERDICT    %s" % ("usable" if self.ok else "NOT usable"))
        return out


def _reject_word(reason: str) -> str:
    """The first two words of a rejection, for the histogram. A rejection
    reason is a sentence on purpose; a tally needs a key."""
    return " ".join(reason.split()[:2]).rstrip(":,")


def summarise(report: EnrolmentReport, samples: Sequence[Sample],
              embs: Sequence, embed_ms: Sequence[float],
              identity_min: float) -> None:
    """Fill the distributions. Split out so a caller that assembled samples
    some other way -- a re-scoring of an existing gallery, say -- gets the
    identical arithmetic rather than a second version of it."""
    kept = [s for s in samples if s.accepted]
    report.offered = len(samples)
    report.accepted = len(kept)
    report.rejected = len(samples) - len(kept)
    report.identity_min = float(identity_min)
    conf = [s.conf for s in kept]
    report.conf_min = min(conf) if conf else 0.0
    report.conf_p50 = _pct(conf, 0.5)
    face = [s.face_px for s in kept]
    report.face_px_min = min(face) if face else 0.0
    report.face_px_p50 = _pct(face, 0.5)
    report.face_px_max = max(face) if face else 0.0
    eyes = [s.eye_px for s in kept]
    report.eye_px_min = min(eyes) if eyes else 0.0
    report.eye_px_p50 = _pct(eyes, 0.5)
    yaws = [s.yaw_deg for s in kept]
    report.yaw_min = min(yaws) if yaws else 0.0
    report.yaw_p50 = _pct(yaws, 0.5)
    report.yaw_max = max(yaws) if yaws else 0.0
    report.yaw_spread = (max(yaws) - min(yaws)) if yaws else 0.0
    report.frontal = sum(1 for y in yaws if abs(y) <= FRONTAL_MAX_DEG)
    report.offaxis = sum(1 for y in yaws if abs(y) >= OFFAXIS_MIN_DEG)
    rolls = [s.roll_deg for s in kept]
    report.roll_min = min(rolls) if rolls else 0.0
    report.roll_max = max(rolls) if rolls else 0.0
    sharp = [s.sharpness for s in kept]
    report.sharp_min = min(sharp) if sharp else 0.0
    report.sharp_p50 = _pct(sharp, 0.5)
    report.embed_ms_p50 = _pct(list(embed_ms), 0.5)
    pairs = pairwise_cosines(embs)
    report.pairs = len(pairs)
    report.cos_min = min(pairs) if pairs else 0.0
    report.cos_p05 = _pct(pairs, 0.05)
    report.cos_p50 = _pct(pairs, 0.5)
    report.cos_max = max(pairs) if pairs else 0.0
    coh = cohesion(embs)
    report.cohesion_min = min(coh) if coh else 0.0
    report.cohesion_p50 = _pct(coh, 0.5)
    tally: dict = {}
    for s in samples:
        if s.accepted:
            continue
        key = _reject_word(s.reason)
        tally[key] = tally.get(key, 0) + 1
    report.rejects = tuple(sorted(tally.items(), key=lambda kv: -kv[1]))
    report.samples = tuple(samples)
    report.checks = judge_gallery(samples, embs, identity_min)


# -------------------------------------------------------------- the session
class EnrolmentSession:
    """Judge frames, embed the ones that pass, and hold nothing else.

    The seams are the same three the rest of this lane uses: a ``detector``
    with ``input_size`` and ``detect(frame)``, a ``recogniser`` with
    ``embed(frame, row)``, and a ``lens`` for the pixel-to-angle map. Tests
    pass stubs; nothing here opens a device or imports cv2.

    ``offer()`` is the whole contract: one frame in, one ``Sample`` out, and
    an embedding added to the gallery ONLY if that sample was accepted. The
    frame is not retained, the crop lives inside the recogniser, and the
    embedding goes straight into the pool.
    """

    def __init__(self, gallery: FaceGallery, label: str, lens,
                 detector, recogniser, limits: SampleLimits, *,
                 head: Optional[HeadModel] = None,
                 now: Callable[[], float] = time.monotonic):
        self.gallery = gallery
        self.label = label
        self.lens = lens
        self.detector = detector
        self.recogniser = recogniser
        self.limits = limits
        self.head = head or HeadModel()
        self._now = now
        self.samples: List[Sample] = []
        self.embs: List[np.ndarray] = []
        self.embed_ms: List[float] = []
        self.frames = 0
        dw, dh = getattr(detector, "input_size", (0, 0))
        self.scale_x = self.lens.width_px / float(dw) if dw else 1.0
        self.scale_y = self.lens.height_px / float(dh) if dh else 1.0

    @property
    def kept(self) -> int:
        return len(self.embs)

    def offer(self, frame, station: str = "") -> Optional[Sample]:
        """Detect, judge, and embed only if judged good.

        Returns None when the detector saw NO face at all -- which is not a
        rejected sample, it is a frame with nobody in it, and counting it as
        a rejection would fill the report with noise while he is still
        settling into position.
        """
        self.frames += 1
        try:
            rows = self.detector.detect(frame)
        except Exception as exc:  # noqa: BLE001 - counted, never swallowed
            log.debug("faceenrol: detect failed", exc_info=True)
            return self._record(Sample(
                index=len(self.samples), station=station, faces=0, conf=0.0,
                face_px=0.0, eye_px=0.0, yaw_deg=0.0, roll_deg=0.0,
                bearing_deg=0.0, sharpness=0.0, accepted=False,
                reason="detector raised: %s" % exc))
        rows = [] if rows is None else list(rows)
        if not len(rows):
            return None
        faces = len(rows)
        best_i = int(np.argmax([float(np.asarray(r).ravel()[IDX_SCORE])
                                for r in rows]))
        row = rows[best_i]
        obs = observe(np.asarray(row, dtype=np.float32).ravel(), self.lens,
                      self.scale_x, self.scale_y, self.head)
        sharp = sharpness(frame, obs.x, obs.y, obs.w, obs.h)
        ok, why = judge_sample(obs, faces, sharp, self.limits)
        sample = Sample(index=len(self.samples), station=station, faces=faces,
                        conf=obs.conf, face_px=obs.face_px,
                        eye_px=obs.eye_px, yaw_deg=obs.yaw_deg,
                        roll_deg=obs.roll_deg, bearing_deg=obs.bearing_deg,
                        sharpness=sharp, accepted=False, reason=why)
        if not ok:
            # NO EMBEDDING IS COMPUTED. This is the rule, in code: SFace
            # answers confidently on inputs that are not faces, so an
            # embedding taken from a rejected detection is not a weak
            # opinion, it is a wrong answer that looks right.
            return self._record(sample)

        t0 = self._now()
        try:
            vec = self.recogniser.embed(frame, row)
        except Exception as exc:  # noqa: BLE001
            log.debug("faceenrol: embed failed", exc_info=True)
            return self._record(Sample(
                **{**sample.as_dict(), "accepted": False,
                   "reason": "recogniser refused: %s" % exc}))
        self.embed_ms.append((self._now() - t0) * 1000.0)
        try:
            self.gallery.add(self.label, vec)
        except ValueError as exc:
            # facegallery's own degenerate-vector guard. It has never fired
            # on a real embedding; if it does, that is a finding and it
            # belongs in the report rather than in a traceback.
            return self._record(Sample(
                **{**sample.as_dict(), "accepted": False,
                   "reason": "gallery refused the embedding: %s" % exc}))
        self.embs.append(np.asarray(vec, dtype=np.float32).ravel().copy())
        vec = None
        return self._record(Sample(**{**sample.as_dict(), "accepted": True}))

    def _record(self, sample: Sample) -> Sample:
        self.samples.append(sample)
        return sample

    def report(self, identity_min: float = SFACE_COSINE_SAME
               ) -> EnrolmentReport:
        rep = EnrolmentReport(label=self.label, frames=self.frames,
                              detector_ok=True,
                              min_conf=self.limits.min_conf)
        summarise(rep, self.samples, self.embs, self.embed_ms, identity_min)
        return rep


# ------------------------------------------------------------- the run loop
def run_enrolment(session: EnrolmentSession, source, *,
                  plan: Sequence[Station] = DEFAULT_PLAN,
                  say: Callable[[str], None] = print,
                  wait: Optional[Callable[[str], None]] = None,
                  frames_per_station: int = 240,
                  gap_s: float = 0.35,
                  identity_min: float = SFACE_COSINE_SAME,
                  now: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> Tuple:
    """Walk the plan, and return ``(report, stations_run)``.

    ``source`` is cv2.VideoCapture's own ``read() -> (ok, frame)``, which is
    what ``jarvis/camera.FeedSource`` hands over -- so this runs against the
    GATED camera and not around it. A read that comes back False is the
    sensing owner having said no mid-run (the curfew edge, or him saying
    "offline mode"), and the correct response is to stop, not to retry: see
    ``jarvis/camera.py``'s ``_GatedDevice``.

    ``gap_s`` is the minimum spacing between two accepted samples. Without
    it, eight frames at 8 fps are one second of one pose, i.e. eight
    near-duplicates and a gallery that fails the ``variation`` check -- and
    a gallery that fails a check it could have avoided wastes his time
    rather than teaching him anything.
    """
    stations_run = 0
    t_start = now()
    stopped = ""
    for station in plan:
        if stopped:
            break
        say("")
        say("[%s] %s" % (station.key, station.prompt))
        say("    hold it; %d samples wanted in this position."
            % station.samples)
        if wait is not None:
            wait("    press Enter when you are in position: ")
        got = 0
        last = 0.0
        for _ in range(int(frames_per_station)):
            if got >= station.samples:
                break
            ok, frame = source.read()
            if not ok or frame is None:
                stopped = ("the frame source stopped delivering -- sensing "
                           "denied the camera, or the device went away")
                break
            sample = session.offer(frame, station.key)
            frame = None            # the frame does not outlive the loop
            if sample is None:
                continue
            if sample.accepted and not station.wants(sample.yaw_deg):
                # Accepted on quality, but he is not in the position this
                # station asked for. Counted for the station anyway -- the
                # sample is good and the pose distribution is judged at the
                # end -- but said out loud, because "the station you thought
                # you were doing is not the one you did" is exactly the thing
                # a numbers-only workflow has to say.
                say("    (yaw %+.1f is outside this station's %+.0f..%+.0f)"
                    % (sample.yaw_deg, station.yaw_lo, station.yaw_hi))
            say(sample.line())
            if not sample.accepted:
                continue
            got += 1
            last = now()
            while now() - last < gap_s:
                sleep(min(0.05, gap_s))
        if got:
            stations_run += 1
        if got < station.samples and not stopped:
            say("    only %d of %d here -- moving on; the pose spread is "
                "judged at the end, not per station." % (got, station.samples))
    rep = session.report(identity_min)
    rep.stations_run = stations_run
    rep.stations_planned = len(plan)
    rep.seconds = max(0.0, now() - t_start)
    if stopped:
        rep.detector_ok = False
        rep.reason = stopped
    return rep, stations_run


# ------------------------------------------------------------ the disk side
def harden(root: Path) -> dict:
    """0700 on the directory, 0600 on every file under it.

    ``FaceGallery.save`` already creates them that way; this repairs a store
    that predates that, or one a backup restored with a umask. Returns the
    modes it found, as octal integers, so the report can print them -- his
    voiceprint.npz is 0664 today, which is how this class of mistake is
    discovered at all.
    """
    found: dict = {}
    root = Path(root)
    if not root.exists():
        return found
    try:
        found["dir"] = int(oct(os.stat(root).st_mode & 0o777)[2:])
        os.chmod(root, 0o700)
    except OSError:
        log.warning("faceenrol: could not chmod %s", root, exc_info=True)
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not path.is_file():
            continue
        try:
            found[path.name] = int(oct(os.stat(path).st_mode & 0o777)[2:])
            os.chmod(path, 0o600)
        except OSError:
            log.warning("faceenrol: could not chmod %s", path, exc_info=True)
    return found


def backup(gallery: FaceGallery, dest: Path) -> dict:
    """Copy every generation to ``dest`` and VERIFY each copy parses.

    SAY THE TRUE THING ABOUT WHAT THIS BUYS. The generational store is not a
    backup: every generation lives in the same directory on the same disk, so
    one disk failure takes all of them at once. There is no backup system on
    this box -- restic is installed, no repository exists, nothing under
    ``~/.aiws_trainer`` is backed up (docs/vision.md section 5). This command
    is the only thing that puts his enrolment somewhere else, and it is only
    as good as where he points it: a second directory on the same disk is a
    copy, not a backup.

    Verification is the point of the ``read`` below. A backup nobody has read
    back is the shape the voiceprint's ``.corrupt-<date>`` copy had -- it
    existed, it was named like a backup, and it held only the fixtures.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(dest, 0o700)
    out: dict = {"dest": str(dest), "copied": 0, "verified": 0,
                 "generations": [], "samples": 0, "failed": []}
    for gen in gallery.generations():
        src = gallery.path_for(gen)
        target = dest / src.name
        try:
            data = src.read_bytes()
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.chmod(target, 0o600)
            out["copied"] += 1
        except OSError as exc:
            out["failed"].append("%s: %s" % (src.name, exc))
            continue
        check = FaceGallery(root=dest)
        if check.load(generation=gen):
            out["verified"] += 1
            out["generations"].append(gen)
            out["samples"] += check.total()
        else:
            out["failed"].append("%s: the copy did not parse" % target.name)
    out["ok"] = bool(out["verified"] and not out["failed"])
    return out


def restore(gallery: FaceGallery, src: Path, reason: str) -> dict:
    """Load the newest generation from ``src`` and save it as a NEW
    generation of the live gallery.

    Never a file copy over the live store: restoring by overwriting is the
    move that made the 2026-09-02 loss unrecoverable. A restore that turns
    out to be the wrong one is then just another generation to roll back.
    """
    src = Path(src)
    out: dict = {"src": str(src), "restored": 0, "generation": 0,
                 "samples": 0, "reason": ""}
    holding = FaceGallery(root=src)
    if not holding.load():
        out["reason"] = "nothing readable in %s" % src
        return out
    # A restore REPLACES; it does not merge. Adding a backup's embeddings on
    # top of whatever the live pool happens to be holding produces a pool
    # that is two enrolments at once, whose provenance says "restore" and
    # whose contents are half something else -- which is exactly the kind of
    # store nobody can reason about afterwards. The generations underneath
    # are untouched, so the thing being replaced is still on disk.
    gallery.reset()
    for label in holding.labels():
        for emb in holding.embeddings(label):
            gallery.add(label, emb)
            out["restored"] += 1
    if not out["restored"]:
        out["reason"] = "the newest generation in %s holds no embeddings" % src
        return out
    # allow_shrink: a restore is deliberately a step BACKWARDS, and the
    # shrink guard exists to catch the accidental version of that.
    out["generation"] = gallery.save(reason=reason, allow_shrink=True)
    out["samples"] = gallery.total()
    return out
