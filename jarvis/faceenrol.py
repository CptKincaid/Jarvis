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
* **Too loose** -- some member does not cluster with the rest: its median
  cosine to the others is under the "same person" bar, OR it is far below the
  pool's OWN median. Either way it drags every future match toward itself,
  because ``FaceGallery.match`` scores against the pool's BEST member.

  BOTH TERMS ARE NEEDED AND NEITHER IS SUFFICIENT. 0.363 is OpenCV's
  VERIFICATION threshold; a healthy pool's own median is ~0.80, so an
  absolute 0.363 bar only fires below ~0.40 and cannot see an outlier at 0.5
  or 0.6 -- which is inside the 0.66-0.92 band the measured OOD collapse
  occupies. The relative term asks the question the absolute one cannot.
  SAY THE LIMIT: what this catches is an outlier IN a pool. It does not catch
  a pool that is TWO tight clusters at a cross-cosine above the bar -- an
  even split makes every member's median the cross value, so the shape is
  invisible to a per-member statistic. What stands there instead is
  ``MAX_FACES_IN_FRAME`` (a second face in frame is refused before any
  embedding), the one-label rule, and judging the pool that will actually be
  written rather than the batch (see ``EnrolmentSession.pool``).

A NOTE ON EVERY TAKE, which is what he asked for on 2026-09-02: "with a note
on what i am doing in the take". Each accepted sample stores his own words
beside the embedding -- "looking at my phone", "looking away", "with glasses"
-- and ``note_rows`` groups the pool's own per-sample cohesion by those words,
worst first. That turns the one number a bad match gives you ("0.41") into the
one sentence you can act on ("your looking-at-my-phone takes are the thin
ones"). The five default stations each carry a note, so a first enrolment is
never a note-less one, and ``custom_stations`` turns anything else he names
into a station of its own.

AND THE SIDE HE HAS NEVER GIVEN. ``pose_spread`` counts ``abs(yaw)``, so it
cannot tell 13 takes from +2 to +55 deg from 13 spread across both sides --
and his are the first kind: every sample in his enrolment AND in his
verification was a POSITIVE yaw. ``coverage`` counts the two sides
separately, ``coverage_lines`` says which one is empty, and
``missing_stations`` asks the NEXT run for exactly the gap instead of reading
the same five instructions back at him. Nothing recorded falls back to the
five stations, because silence is not evidence of coverage.

Nothing here imports cv2, torch or a model. The detector, the recogniser and
the frame source are all seams, so the whole module runs in the suite with no
camera, no weights, no display and no GPU.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from jarvis.facedetect import PROBE_THRESHOLD
from jarvis.facegallery import (SFACE_COSINE_SAME, FaceGallery, Take,
                                clean_note, cosine)
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
# A cohesion floor is only half a check. 0.363 is OpenCV's VERIFICATION
# threshold -- "are these two the same person, at some FAR" -- and a healthy
# pool's own median sits around 0.80, i.e. 0.44 ABOVE it. So an absolute
# 0.363 bar cannot see an outlier at 0.5 or 0.6, which is squarely inside the
# band the measured OOD collapse occupies (0.66-0.92 for non-face crops). The
# second term asks the question the first cannot: is this member unlike the
# pool RELATIVE to how varied the pool already is. Scaled by the pool's own
# MAD so a genuinely spread enrolment -- his 14-54 deg range -- is not
# punished for being spread, with a fixed margin underneath so a tight pool
# does not get a razor-thin bar out of a near-zero MAD.
COHESION_MARGIN = 0.20
COHESION_MAD_K = 6.0
# Two faces in frame during enrolment is the one thing that must never be
# quietly averaged in: a gallery that learned a visitor identifies the wrong
# person confidently, and nothing about it looks wrong afterwards.
MAX_FACES_IN_FRAME = 1
# What ``camera.min_conf`` ships as. A run under this is not refused -- he may
# have measured his own face and moved it -- but it is SAID, in the report
# header, because a lowered bar is the one setting that quietly turns the
# gate in the module docstring into a formality.
DEFAULT_MIN_CONF = 0.6


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

    def __post_init__(self) -> None:
        """Refuse a bar that is not a bar.

        ``min_conf`` is read from a user-editable key
        (``camera.min_conf`` in ~/.config/jarvis/assistant.json). Set to 0 it
        removes the gate this whole module is built on -- a detection scoring
        0.01 would be embedded and written under his label -- and nothing
        anywhere said so. The floor is the detector's OWN filter
        (``facedetect.PROBE_THRESHOLD``): under it the bar is not merely low,
        it is below the score of anything YuNet will even hand back, so it
        cannot reject anything at all.

        Spelled ``not (x >= floor)`` rather than ``x < floor`` so a NaN in the
        config fails SHUT. ``NaN < 0.3`` is False, which would have let a
        non-comparable bar through the one check that exists to stop it.
        """
        if not (float(self.min_conf) >= float(PROBE_THRESHOLD)):
            raise ValueError(
                "min_conf %r is not a usable detector bar: it must be at "
                "least %.2f, the floor the detector itself filters at. Below "
                "that the confidence gate rejects nothing, and SFace returns "
                "a CONFIDENT match on a crop that is not a face -- so a bar "
                "of 0 does not make enrolment lenient, it makes it wrong."
                % (self.min_conf, float(PROBE_THRESHOLD)))


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
    # WHAT HE WAS DOING, in his own words, stored beside every embedding this
    # station produces. The five below carry one each, so a first enrolment
    # is not a note-less one -- the notes are the thing that lets a bad match
    # six weeks from now be answered with "your looking-at-my-phone takes are
    # the weak ones" instead of a shrug.
    note: str = ""

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
    Station("lens", "Look straight into the camera lens.", -15.0, 15.0, 3,
            note="looking at the lens"),
    Station("screen",
            "Now look at your screen exactly as you do when you are working.",
            18.0, 62.0, 3, note="looking at my screen"),
    Station("across",
            "Turn your head the OTHER way -- as if looking at the far side "
            "of the desk.", -62.0, -18.0, 3,
            note="turned the other way"),
    Station("back",
            "Back to the camera, but sit back -- further away than usual.",
            -15.0, 15.0, 2, note="sitting back from the camera"),
    Station("close",
            "Lean in closer than usual, still looking at the camera.",
            -15.0, 15.0, 2, note="leaning in close"),
)
_BY_KEY = {s.key: s for s in DEFAULT_PLAN}


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
    # Defaulted, and last, so every existing construction of this dataclass
    # -- including the ``Sample(**{**sample.as_dict(), ...})`` rebuilds in
    # ``offer`` -- keeps working unchanged.
    note: str = ""

    def as_dict(self) -> dict:
        return {"index": self.index, "station": self.station,
                "faces": self.faces, "conf": self.conf,
                "face_px": self.face_px, "eye_px": self.eye_px,
                "yaw_deg": self.yaw_deg, "roll_deg": self.roll_deg,
                "bearing_deg": self.bearing_deg,
                "sharpness": self.sharpness, "accepted": self.accepted,
                "reason": self.reason, "note": self.note}

    def line(self) -> str:
        return ("  %2d %-7s %s conf %.2f  face %4.0fpx  eyes %3.0fpx  "
                "yaw %+5.1f  roll %+5.1f  sharp %.3f  %s%s"
                % (self.index, self.station,
                   "KEEP" if self.accepted else "drop", self.conf,
                   self.face_px, self.eye_px, self.yaw_deg, self.roll_deg,
                   self.sharpness, self.reason,
                   ("  [%s]" % self.note) if self.note else ""))


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

    EVERY BAR IS SPELLED ``not (value PASSES)`` RATHER THAN ``value FAILS``,
    and that is not a style choice. ``NaN < 0.6`` is False, so the natural
    spelling makes a non-finite score clear the bar -- and then clear every
    bar under it, since they are all comparisons too. The one gate the whole
    safety argument rests on would fail OPEN on exactly the input least
    likely to be a face. Inverted, a NaN passes nothing.
    """
    if faces > int(limits.max_faces):
        return False, ("%d faces in frame -- a gallery that learns a visitor "
                       "identifies the wrong person confidently" % faces)
    if not (obs.conf >= limits.min_conf):
        return False, ("conf %.2f under the %.2f detector bar -- no "
                       "embedding is computed from this"
                       % (obs.conf, limits.min_conf))
    if not obs.landmarks_ok:
        return False, "the eye landmarks coincide; the crop cannot be aligned"
    if not (obs.face_px >= limits.min_face_px):
        return False, ("face %.0f px under SFace's %.0f px input -- the crop "
                       "would be upsampled" % (obs.face_px,
                                               limits.min_face_px))
    if not (obs.eye_px >= limits.min_eye_px):
        return False, ("interocular %.0f px under %.0f -- too small to align"
                       % (obs.eye_px, limits.min_eye_px))
    if not (abs(obs.yaw_deg) <= limits.max_yaw_deg):
        return False, ("yaw %+.1f deg past the %.0f deg limit"
                       % (obs.yaw_deg, limits.max_yaw_deg))
    if not (abs(obs.roll_deg) <= limits.max_roll_deg):
        return False, ("roll %+.1f deg past the %.0f deg limit"
                       % (obs.roll_deg, limits.max_roll_deg))
    if not (sharp >= limits.min_sharpness):
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


# --------------------------------------------- what the takes say they were
# The label a take with no note is printed under. A count under a name is a
# thing he can act on; a blank cell is a thing he ignores.
NO_NOTE = "(no note)"
# How many takes a side wants before it stops being a gap. Two, the same
# number ``MIN_FRONTAL`` and ``MIN_OFFAXIS`` already use, because one sample
# of a pose is one sample of one instant of that pose.
COVERAGE_WANT = 2


@dataclass(frozen=True)
class NoteRow:
    """One pose, as a row he can read: how many takes carry that note, and
    how well they cohere with the rest of the pool."""

    note: str
    count: int
    cohesion_p50: float
    cohesion_min: float
    yaw_min: float
    yaw_max: float
    recorded: int

    def as_tuple(self) -> tuple:
        return (self.note, self.count, self.cohesion_p50, self.cohesion_min,
                self.yaw_min, self.yaw_max, self.recorded)

    def line(self) -> str:
        yaw = ("yaw %+5.1f..%+5.1f" % (self.yaw_min, self.yaw_max)
               if self.recorded else "no angle recorded")
        return ("  %-28s %2d takes  cohesion p50 %.3f  worst %.3f  %s"
                % (self.note[:28], self.count, self.cohesion_p50,
                   self.cohesion_min, yaw))


def note_rows(embs: Sequence, takes: Sequence[Take]) -> Tuple[NoteRow, ...]:
    """The pool grouped by what he said he was doing, WEAKEST FIRST.

    THIS IS THE WHOLE POINT OF STORING A NOTE. 128 floats cannot answer "why
    did it not know me just then"; a note can, because the pool's own
    cohesion is already computed per sample (``cohesion`` above) and grouping
    those numbers by note turns "the gallery medians 0.62" into "your
    looking-at-my-phone takes median 0.41 and everything else medians 0.78".

    Sorted ascending by median cohesion so the first row is the answer to the
    question he will actually ask. Ties break on the smaller group, because a
    weak pose with two takes is a thinner claim than a weak pose with eight.

    SAY THE LIMIT. A low row is not proof that pose is bad -- a genuinely
    distinct pose SHOULD cohere less with a frontal pool, which is exactly
    the spread the enrolment asks for. What the row tells him is where the
    pool is thin, which is where to add takes; the absolute floor for "this
    is not the same person at all" is the cohesion check, not this."""
    coh = cohesion(list(embs))
    takes = list(takes)
    groups: dict = {}
    for i, take in enumerate(takes[:len(coh)]):
        key = take.note or NO_NOTE
        groups.setdefault(key, []).append(i)
    rows = []
    for note, idx in groups.items():
        scores = [coh[i] for i in idx]
        yaws = [float(takes[i].yaw_deg) for i in idx
                if takes[i].yaw_deg is not None]
        rows.append(NoteRow(note=note, count=len(idx),
                            cohesion_p50=_pct(scores, 0.5),
                            cohesion_min=min(scores) if scores else 0.0,
                            yaw_min=min(yaws) if yaws else 0.0,
                            yaw_max=max(yaws) if yaws else 0.0,
                            recorded=len(yaws)))
    rows.sort(key=lambda r: (r.cohesion_p50, r.count))
    return tuple(rows)


def weakest_note(rows: Sequence[NoteRow]) -> str:
    """The note of the weakest group, or "" when there is nothing to say.

    A single group is not a comparison, so it is not an answer: with one
    note there is no "worst" pose, there is just the pool."""
    rows = [r for r in rows if r.note != NO_NOTE]
    return rows[0].note if len(rows) > 1 else ""


def coverage(takes: Sequence[Take]) -> dict:
    """How many takes sit on each SIDE of the lens, and how many say nothing.

    WHY THE SIDES ARE COUNTED SEPARATELY, which ``pose_spread`` does not do.
    That check counts ``abs(yaw)``, so 13 takes spread from +2 to +55 deg
    look identical to 13 spread from -55 to +55 -- and his are the first
    kind: every sample in his enrolment AND in his verification carried a
    POSITIVE yaw. He has no coverage at all on the other side, and no number
    the existing report prints says so.

    ``unrecorded`` is its own count and is NEVER folded into frontal. His
    live generation carries no angles, and treating "not measured" as "0 deg"
    would report thirteen perfectly frontal takes he never gave. A yaw
    between ``FRONTAL_MAX_DEG`` and ``OFFAXIS_MIN_DEG`` lands in no bucket on
    purpose: that band is the existing dead zone between "at the lens" and
    "at the screen", and inventing a third name for it would make the counts
    stop adding up to the thing they are compared against."""
    out = {"recorded": 0, "unrecorded": 0, "negative": 0, "frontal": 0,
           "positive": 0, "yaw_min": 0.0, "yaw_max": 0.0, "total": 0}
    yaws: List[float] = []
    for take in takes:
        out["total"] += 1
        if take.yaw_deg is None:
            out["unrecorded"] += 1
            continue
        out["recorded"] += 1
        yaw = float(take.yaw_deg)
        yaws.append(yaw)
        if yaw <= -OFFAXIS_MIN_DEG:
            out["negative"] += 1
        elif abs(yaw) <= FRONTAL_MAX_DEG:
            out["frontal"] += 1
        elif yaw >= OFFAXIS_MIN_DEG:
            out["positive"] += 1
    if yaws:
        out["yaw_min"], out["yaw_max"] = min(yaws), max(yaws)
    return out


def coverage_lines(cov: dict) -> List[str]:
    """The coverage as he should read it, including the gap."""
    out = ["coverage   %d frontal (|yaw| <= %.0f), %d toward the screen "
           "(yaw >= +%.0f), %d turned the other way (yaw <= -%.0f); "
           "%d with no pose record"
           % (cov["frontal"], FRONTAL_MAX_DEG, cov["positive"],
              OFFAXIS_MIN_DEG, cov["negative"], OFFAXIS_MIN_DEG,
              cov["unrecorded"])]
    if cov["recorded"] and not cov["negative"]:
        out.append(
            "NOTE       no takes at all turned the OTHER way (negative yaw). "
            "Measured on his camera 2026-09-02, every sample of both his "
            "enrolment and his verification was a POSITIVE yaw -- so a "
            "gallery like this has never seen that side of his face and "
            "will fail on it silently. Run again with the 'across' station, "
            "or --pose \"turned the other way\".")
    if cov["recorded"] and not cov["positive"]:
        out.append(
            "NOTE       no takes toward the screen (positive yaw), which is "
            "the pose he is in most of the working day.")
    if cov["unrecorded"] and not cov["recorded"]:
        out.append(
            "NOTE       not one take carries a pose record, so this gallery "
            "cannot say which poses it covers. That is what his generation "
            "1 is: 13 embeddings written before takes were recorded. Enrol "
            "again, or --append, to start recording them.")
    return out


def missing_stations(takes: Sequence[Take],
                     want: int = COVERAGE_WANT) -> Tuple[Station, ...]:
    """Ask for what the gallery is MISSING, rather than for the list again.

    Better than another fixed script, and it is the difference between a tool
    that repeats itself and one that reads what is already there: if he has
    six frontal takes and six at his screen, the only thing worth another
    minute of his time is the side he has never given.

    NOTHING RECORDED FALLS BACK TO THE FIVE. A gallery that says nothing
    about its poses supports no inference at all -- and the five stations
    were chosen against his measured geometry (~+6 deg at the lens, ~+55 at
    his screen), so they are the right thing to run when there is nothing to
    reason from. Silence is not evidence of coverage."""
    cov = coverage(takes)
    if not cov["recorded"]:
        return DEFAULT_PLAN
    out: List[Station] = []
    for key, have in (("lens", cov["frontal"]),
                      ("screen", cov["positive"]),
                      ("across", cov["negative"])):
        if have < int(want):
            out.append(replace(_BY_KEY[key], samples=int(want) - have))
    return tuple(out)


def _slug(text: str) -> str:
    """A short key for the report's station column, from his own words.

    The column is 7 characters wide and holds a name he has to recognise at a
    glance, so the longest word that is not a filler wins -- "looking at my
    phone" is "phone", not "looking"."""
    stop = {"a", "an", "the", "at", "in", "on", "my", "me", "with", "to",
            "of", "and", "is", "am", "looking", "look", "while", "when"}
    plain = "".join(c.lower() if (c.isalnum() or c.isspace()) else " "
                    for c in str(text))
    words = plain.split()
    keep = [w for w in words if w not in stop] or words
    return (keep[0][:7] if keep else "take")


def custom_stations(poses: Sequence[str],
                    samples: int = 3) -> Tuple[Station, ...]:
    """One station per take he named, in his own words.

    A NAMED TAKE JUDGES NO HEAD ANGLE. The five default stations have yaw
    windows because they were written against his measured geometry; "with my
    glasses off" and "looking at my phone" make no claim about yaw at all, so
    the window is the full range the sample gate already allows and the
    station never tells him he is in the wrong position. What it does instead
    is record what he said, which is the thing he asked for.

    An empty pose is a ValueError rather than a blank note: a take stored
    under "" is indistinguishable from the takes that predate notes, so it
    would silently become part of the "no pose record" count."""
    out: List[Station] = []
    used: set = set()
    for i, text in enumerate(poses):
        note = clean_note(text)
        if not note:
            raise ValueError(
                "pose %d is empty. A take needs words on it -- that is the "
                "whole point of naming one." % (i + 1))
        key = _slug(note)
        while key in used:
            key = ("%s%d" % (key[:6], i + 1))[:7]
        used.add(key)
        out.append(Station(key=key,
                           prompt="Now: %s. Hold it." % note,
                           yaw_lo=-MAX_YAW_DEG, yaw_hi=MAX_YAW_DEG,
                           samples=int(samples), note=note))
    return tuple(out)


def choose_plan(takes: Sequence[Take], poses: Sequence[str] = (),
                pose_samples: int = 3,
                mode: str = "auto") -> Tuple[Tuple[Station, ...], str]:
    """``(plan, why)`` -- and the ``why`` is printed, because a run that
    quietly did something other than the five stations is a run whose numbers
    he will misread later."""
    if poses:
        named = custom_stations(poses, pose_samples)
        why = ("the %d take%s you named"
               % (len(poses), "" if len(poses) == 1 else "s"))
        if mode == "full":
            # --plan full --pose X: the five stations AND the named take.
            # This is the run that can pass when the named take alone
            # cannot -- a first enrolment, or an append onto takes that
            # carry no angle (see ``plan_shortfalls``) -- and it is what
            # the hand-over in jarvis/enrolentry.py writes in those cases.
            # Named takes come LAST so the report reads in the order he
            # will do them.
            return (DEFAULT_PLAN + named,
                    "the full five-station script plus %s" % why)
        return named, why
    if mode == "full":
        return DEFAULT_PLAN, "the full five-station script (--plan full)"
    gaps = missing_stations(takes)
    cov = coverage(takes)
    if not cov["recorded"]:
        if mode == "missing":
            return (), ("there is no recorded coverage to be missing from. "
                        "The gap is measured against the takes this run "
                        "KEEPS, and a run without --append replaces them")
        if not cov["total"]:
            return (DEFAULT_PLAN,
                    "the full five-station script -- this run replaces the "
                    "pool, so it has to stand on its own")
        return (DEFAULT_PLAN,
                "the full five-station script -- %d stored take%s carry no "
                "pose record, so there is nothing to reason from"
                % (cov["total"], "" if cov["total"] == 1 else "s"))
    if not gaps:
        if mode == "missing":
            return (), ("nothing is missing: %d frontal, %d toward the "
                        "screen, %d the other way, all at or above %d"
                        % (cov["frontal"], cov["positive"], cov["negative"],
                           COVERAGE_WANT))
        return (DEFAULT_PLAN,
                "the full five-station script -- your recorded coverage is "
                "already complete, so this is a refresh rather than a gap")
    return (gaps, "the %d station%s your gallery is missing"
            % (len(gaps), "" if len(gaps) == 1 else "s"))


def _station_options(station: Station) -> List[Tuple[int, int, float, float]]:
    """The ways one station's samples can land, as ``(frontal, off_axis,
    yaw_lo, yaw_hi)`` -- every sample in one band, and the angles the
    station can reach for the spread.

    A station whose window reaches ONE band -- the five defaults -- puts
    every sample there and may spread across its whole window. A station
    whose window reaches BOTH bands is a named take, and a named take is
    ONE pose: "looking at my phone" is not a head sweep, so its samples
    land together, in one band, at one angle. It is offered each band it
    can reach and never both at once, which is what makes three takes of one
    named pose unable to be "2 frontal AND 2 off-axis" however many are
    captured. A window that reaches neither band (the dead zone between
    ``FRONTAL_MAX_DEG`` and ``OFFAXIS_MIN_DEG``) counts for nothing, exactly
    as ``judge_gallery`` would count it."""
    lo, hi = float(station.yaw_lo), float(station.yaw_hi)
    n = int(station.samples)
    frontal = lo <= FRONTAL_MAX_DEG and hi >= -FRONTAL_MAX_DEG
    offaxis = hi >= OFFAXIS_MIN_DEG or lo <= -OFFAXIS_MIN_DEG
    if frontal and offaxis:
        at = min(max(0.0, lo), hi)          # the nearest-to-straight angle
        out = [(n, 0, at, at)]
        if hi >= OFFAXIS_MIN_DEG:
            out.append((0, n, hi, hi))
        if lo <= -OFFAXIS_MIN_DEG:
            out.append((0, n, lo, lo))
        return out
    if frontal:
        return [(n, 0, lo, hi)]
    if offaxis:
        return [(0, n, lo, hi)]
    return [(0, 0, lo, hi)]


def plan_shortfalls(takes: Sequence[Take],
                    plan: Sequence[Station]) -> Tuple[str, ...]:
    """The checks in ``judge_gallery`` this plan CANNOT pass, one line each
    -- or ``()`` when the arithmetic allows it to.

    ARITHMETIC, NOT A PREDICTION. Two of the four checks are decided by
    counting before a single frame exists: ``samples`` is the size of the
    pool the run would save, and ``pose_spread`` is how many angles land in
    each band and how far apart they are. Both are knowable from the plan
    and the takes it keeps, so a run that fails them is known to fail before
    he sits down -- and the minute in front of the lens is his. ``variation``
    and ``cohesion`` are about the vectors and are not guessed at here.

    ``takes`` are the takes this run KEEPS: the stored ones under --append,
    nothing at all otherwise (a plain run replaces the pool). A kept take
    with no recorded angle counts toward ``samples`` and toward nothing
    else, which is the rule ``judge_gallery`` applies to it -- and it is the
    whole reason this exists: his live generation 1 is 13 embeddings that
    carry no angle, so appending three takes of one named pose onto it puts
    16 in the pool, clears the floor, and can never satisfy the spread. The
    script printed "--append is almost certainly what you want" over exactly
    that run (F16, reproduced 2026-09-03), and the hand-over for a FIRST
    enrolment with a named pose announced one station of three takes as a
    valid run against a floor of eight (F34).

    Best case is taken throughout: each station is allowed to land wherever
    its window lets it, the way ``_station_options`` spells out, and the
    plan is short only if NO way of landing passes. The search is a small
    dynamic programme over (frontal so far, off-axis so far, lowest angle,
    highest angle), capped at the wanted counts, so a plan of any realistic
    length costs microseconds."""
    stations = list(plan)
    wanted = sum(int(st.samples) for st in stations)
    kept = list(takes)
    pool = len(kept) + wanted
    out: List[str] = []
    if pool < MIN_SAMPLES:
        out.append("samples: %d in the pool this run would save (%d kept + "
                   "%d wanted) against a floor of %d"
                   % (pool, len(kept), wanted, MIN_SAMPLES))
    cov = coverage(kept)
    yaws = [float(t.yaw_deg) for t in kept if t.yaw_deg is not None]
    have_f = min(cov["frontal"], MIN_FRONTAL)
    have_o = min(cov["positive"] + cov["negative"], MIN_OFFAXIS)
    lo0 = min(yaws) if yaws else None
    hi0 = max(yaws) if yaws else None
    states = {(have_f, have_o, lo0, hi0)}
    for st in stations:
        nxt = set()
        for f, o, lo, hi in states:
            for df, do, slo, shi in _station_options(st):
                nxt.add((min(f + df, MIN_FRONTAL), min(o + do, MIN_OFFAXIS),
                         slo if lo is None else min(lo, slo),
                         shi if hi is None else max(hi, shi)))
        states = nxt

    def spread_of(s) -> float:
        return 0.0 if s[2] is None else float(s[3] - s[2])

    if not any(s[0] >= MIN_FRONTAL and s[1] >= MIN_OFFAXIS
               and spread_of(s) >= MIN_YAW_SPREAD_DEG for s in states):
        best = max(states, key=lambda s: (s[0] + s[1], spread_of(s)))
        blank = int(cov["unrecorded"])
        out.append("pose_spread: at best %d frontal (want %d) and %d off-axis "
                   "(want %d) over %.0f deg of spread (want %.0f)%s"
                   % (best[0], MIN_FRONTAL, best[1], MIN_OFFAXIS,
                      spread_of(best), MIN_YAW_SPREAD_DEG,
                      (" -- %d kept take(s) carry no angle and count for "
                       "nothing here" % blank) if blank else ""))
    return tuple(out)


def member_name(index: int, stored: int, kept: Sequence[Sample]) -> str:
    """What to call pool member ``index`` in a line he will paste.

    The pool is what was already in the gallery followed by what this run
    added, so an index below ``stored`` is a member from an earlier
    generation that this run never saw -- and calling it "#3" would point him
    at a sample line in THIS report that has nothing to do with it. The
    numbering of this run's members is the SAMPLE index, not the embedding
    index, because the sample lines are numbered over every candidate
    including the dropped ones.
    """
    if index < stored:
        return ("stored embedding %d of the %d already in the gallery (from "
                "an earlier generation, not captured in this run)"
                % (index + 1, stored))
    j = index - stored
    return "#%d" % (kept[j].index if j < len(kept) else j)


def judge_gallery(samples: Sequence[Sample], embs: Sequence,
                  identity_min: float = SFACE_COSINE_SAME,
                  min_samples: int = MIN_SAMPLES,
                  pool: Optional[Sequence] = None,
                  takes: Optional[Sequence[Take]] = None) -> Tuple[Check, ...]:
    """The pass/fail lines that decide whether this gallery may be saved.

    A gallery that is too tight fails him in every pose but one; a gallery
    that is too loose has somebody else in it. Both are worse than no gallery,
    because both are SILENT -- and the second is worse than the first,
    because it is confident.

    ``pool`` IS THE SET THAT WILL BE WRITTEN, and it is what gets judged --
    ``embs`` is only what this run captured. They differ under ``--append``,
    which loads the existing gallery first and saves loaded+new: judging the
    batch there meant a whole appended batch of a DIFFERENT PERSON passed
    every check and was written under his label, while re-running these same
    checks over the 26 embeddings that actually landed said [FAIL] cohesion.

    THE POSE SPREAD IS JUDGED OVER RECORDED ANGLES, WHICH IS A CHANGE, AND
    THE OLD REASON FOR NOT DOING IT HAS GONE. It used to be a statement
    about THIS RUN alone, with the stated reason that "no yaw is stored with
    an embedding -- so an append has to earn the spread again rather than
    inherit a claim nothing can verify". Yaw IS stored now (``Take.yaw_deg``,
    written since the notes landed), so the claim is verifiable for every
    take that carries one, and refusing to look at it had a real cost: an
    append that runs ONLY the station his gallery is missing -- the whole
    point of ``missing_stations`` -- covers one pose by definition and could
    never pass a spread computed from the run alone. It would have made the
    feature that reads his gaps unusable.

    A take with NO recorded angle still contributes NOTHING, which keeps the
    old guarantee exactly where the old reason still applies: appending onto
    his generation 1 -- 13 embeddings written before takes were recorded --
    earns the spread from this run or not at all, because there is no
    evidence to inherit. Evidence is used where it exists and assumed
    nowhere.
    """
    kept = [s for s in samples if s.accepted]
    pool = list(embs) if pool is None else list(pool)
    # The run's embeddings are appended to whatever was loaded, so anything
    # before them is stored history. If the pool somehow does not contain
    # this run, fall back to the run rather than name the wrong members.
    stored = len(pool) - len(embs)
    if stored < 0:
        pool, stored = list(embs), 0
    checks: List[Check] = [
        Check("samples", len(pool) >= int(min_samples),
              "%d embeddings to be saved (%d from this run's %d candidate "
              "frames, %d already in the gallery), against a floor of %d"
              % (len(pool), len(embs), len(samples), stored,
                 int(min_samples)))]
    if not pool:
        checks.append(Check("pose_spread", None, "no samples to measure"))
        checks.append(Check("variation", None, "no samples to compare"))
        checks.append(Check("cohesion", None, "no samples to compare"))
        return tuple(checks)

    run_yaws = [s.yaw_deg for s in kept]
    stored_takes = list(takes or ())[:stored]
    kept_yaws = [float(t.yaw_deg) for t in stored_takes
                 if t.yaw_deg is not None]
    yaws = kept_yaws + run_yaws
    spread = (max(yaws) - min(yaws)) if yaws else 0.0
    frontal = sum(1 for y in yaws if abs(y) <= FRONTAL_MAX_DEG)
    offaxis = sum(1 for y in yaws if abs(y) >= OFFAXIS_MIN_DEG)
    checks.append(Check(
        "pose_spread",
        (spread >= MIN_YAW_SPREAD_DEG and frontal >= MIN_FRONTAL
         and offaxis >= MIN_OFFAXIS),
        "yaw %+.1f..%+.1f deg (%.1f deg of spread, want %.0f), %d frontal "
        "(<=%.0f deg, want %d), %d off-axis (>=%.0f deg, want %d), over %d "
        "angle(s) from this run plus %d recorded with earlier takes (%d "
        "stored take(s) carry no angle and count for nothing). He is at "
        "~14 deg looking at the lens and ~54 at his screen, so a gallery "
        "without both fails the moment he turns to work."
        % (min(yaws) if yaws else 0.0, max(yaws) if yaws else 0.0, spread,
           MIN_YAW_SPREAD_DEG, frontal, FRONTAL_MAX_DEG, MIN_FRONTAL,
           offaxis, OFFAXIS_MIN_DEG, MIN_OFFAXIS, len(run_yaws),
           len(kept_yaws), len(stored_takes) - len(kept_yaws))))

    pairs = pairwise_cosines(pool)
    p50 = _pct(pairs, 0.5)
    checks.append(Check(
        "variation", (not pairs) or p50 <= NEAR_DUPLICATE_MAX,
        "%d pairs, cosine min %.3f p50 %.3f max %.3f. Above %.2f at the "
        "median the samples are one face repeated and carry no more "
        "information than the first of them."
        % (len(pairs), min(pairs) if pairs else 0.0, p50,
           max(pairs) if pairs else 0.0, NEAR_DUPLICATE_MAX)))

    coh = cohesion(pool)
    worst_i = min(range(len(coh)), key=lambda i: coh[i]) if coh else 0
    worst = min(coh) if coh else 0.0
    coh_p50 = _pct(coh, 0.5)
    # MAD, not the standard deviation: the outlier being looked for is IN the
    # sample, and one member cannot move a median absolute deviation the way
    # it moves a mean one.
    mad = _pct([abs(c - coh_p50) for c in coh], 0.5) if coh else 0.0
    margin = max(COHESION_MARGIN, COHESION_MAD_K * mad)
    relative_floor = coh_p50 - margin
    # Both bars, and both spelled so a NaN fails SHUT.
    ok = (len(coh) < 2) or (worst >= float(identity_min)
                            and worst >= relative_floor)
    checks.append(Check(
        "cohesion", ok,
        "worst member is %s at median cosine %.3f to the others, against "
        "OpenCV's %.3f 'same person' bar for SFace AND a relative floor of "
        "%.3f (the pool's own median %.3f less a %.3f margin, %.3f x its "
        "%.3f MAD). A member under the absolute bar is a different person or "
        "a crop that is not a face; a member under the relative floor is "
        "unlike this pool by this pool's own standard, which is the band the "
        "absolute bar cannot see -- a healthy pool medians near 0.80 and "
        "0.363 sits 0.44 below it. match() scores against the pool's BEST "
        "member, so one is enough."
        % (member_name(worst_i, stored, kept), worst, float(identity_min),
           relative_floor, coh_p50, margin, COHESION_MAD_K, mad)))
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
    # The pool the checks were run over and that save() would write. Equal to
    # ``accepted`` on a fresh run; larger under --append, where the numbers
    # above describe the MERGED gallery and not just this run's batch.
    pool_total: int = 0
    pool_stored: int = 0
    # What the pool says its takes were, and where it is thin. Both are read
    # off the SAVED pool, not off this run's batch, for the same reason the
    # cosines are: the numbers printed have to be the numbers judged.
    cov_frontal: int = 0
    cov_positive: int = 0
    cov_negative: int = 0
    cov_unrecorded: int = 0
    weakest: str = ""
    notes: tuple = ()            # NoteRow, worst cohesion first
    coverage_notes: tuple = ()   # the prose lines for the gaps
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
               if k not in ("checks", "samples", "rejects", "notes",
                            "coverage_notes")}
        out["ok"] = self.ok
        out["checks"] = [list(c.as_tuple()) for c in self.checks]
        out["samples"] = [s.as_dict() for s in self.samples]
        out["rejects"] = [[str(k), int(v)] for k, v in self.rejects]
        out["notes"] = [list(r.as_tuple()) for r in self.notes]
        out["coverage_notes"] = [str(x) for x in self.coverage_notes]
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
            "pool       %d embeddings judged and to be saved (%d captured "
            "now, %d already in the gallery)"
            % (self.pool_total, self.pool_total - self.pool_stored,
               self.pool_stored),
            "pairwise   %d pairs  cosine min %.3f p05 %.3f p50 %.3f max %.3f"
            % (self.pairs, self.cos_min, self.cos_p05, self.cos_p50,
               self.cos_max),
            "cohesion   worst %.3f  p50 %.3f  against the %.3f same-person "
            "bar" % (self.cohesion_min, self.cohesion_p50,
                     self.identity_min),
        ]
        out.extend(coverage_lines({"frontal": self.cov_frontal,
                                   "positive": self.cov_positive,
                                   "negative": self.cov_negative,
                                   "unrecorded": self.cov_unrecorded,
                                   "recorded": (self.cov_frontal
                                                + self.cov_positive
                                                + self.cov_negative),
                                   "total": self.pool_total}))
        if self.notes:
            out.append("by take    what each pose is worth, weakest first. "
                       "This is what the notes are FOR: when a match scores "
                       "badly, the first row is the answer.")
            out.extend(r.line() for r in self.notes)
        if self.weakest:
            out.append("WEAKEST    the %r takes -- that is where to add "
                       "more, not to the pose that is already strong."
                       % self.weakest)
        if self.min_conf < DEFAULT_MIN_CONF:
            # The one setting that can turn the gate this module is built on
            # into a formality, said where he cannot miss it rather than left
            # to be inferred from the header's bar figure.
            out.append(
                "NOTE       the detector bar is %.2f, LOWERED from the %.2f "
                "default. That bar is the only thing between a crop that is "
                "not a face and this gallery -- SFace matches non-face crops "
                "CONFIDENTLY (0.66-0.92)." % (self.min_conf,
                                              DEFAULT_MIN_CONF))
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
              identity_min: float, pool: Optional[Sequence] = None,
              takes: Optional[Sequence[Take]] = None) -> None:
    """Fill the distributions. Split out so a caller that assembled samples
    some other way -- a re-scoring of an existing gallery, say -- gets the
    identical arithmetic rather than a second version of it.

    ``pool`` is what will be SAVED (see ``judge_gallery``); the cosine and
    cohesion figures are computed over it, so the numbers printed are the
    numbers judged. Under ``--append`` that is the merged gallery and not
    this run's batch -- the per-sample lines still come from this run,
    because those are the frames he was in front of.
    """
    kept = [s for s in samples if s.accepted]
    pool = list(embs) if pool is None else list(pool)
    if len(pool) < len(embs):
        pool = list(embs)
    report.pool_total = len(pool)
    report.pool_stored = len(pool) - len(embs)
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
    pairs = pairwise_cosines(pool)
    report.pairs = len(pairs)
    report.cos_min = min(pairs) if pairs else 0.0
    report.cos_p05 = _pct(pairs, 0.05)
    report.cos_p50 = _pct(pairs, 0.5)
    report.cos_max = max(pairs) if pairs else 0.0
    coh = cohesion(pool)
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
    # The takes belong to the POOL, so a caller that could not supply them
    # (a re-scoring of raw vectors) gets one blank take per member rather
    # than a mismatch -- which reads as "no pose record", which is true.
    tks = list(takes) if takes is not None else []
    if len(tks) < len(pool):
        tks = tks + [Take()] * (len(pool) - len(tks))
    tks = tks[:len(pool)]
    report.checks = judge_gallery(samples, embs, identity_min, pool=pool,
                                  takes=tks)
    cov = coverage(tks)
    report.cov_frontal = cov["frontal"]
    report.cov_positive = cov["positive"]
    report.cov_negative = cov["negative"]
    report.cov_unrecorded = cov["unrecorded"]
    report.coverage_notes = tuple(coverage_lines(cov)[1:])
    report.notes = note_rows(pool, tks)
    report.weakest = weakest_note(report.notes)


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

    def offer(self, frame, station: str = "",
              note: str = "") -> Optional[Sample]:
        """Detect, judge, and embed only if judged good.

        ``note`` is what he said he was doing -- "looking at my phone" -- and
        it is stored beside the embedding, never instead of anything. It has
        no effect on whether the sample is accepted: a note is a label on a
        measurement, not evidence about it.

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
                reason="detector raised: %s" % exc, note=note))
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
                        sharpness=sharp, accepted=False, reason=why,
                        note=note)
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
            self.gallery.add(self.label, vec, note=note,
                             yaw_deg=obs.yaw_deg)
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

    def pool(self) -> List[np.ndarray]:
        """The embeddings ``gallery.save()`` would write for this label:
        whatever was loaded before the run, PLUS what this run added.

        THIS, NOT ``self.embs``, IS WHAT MUST BE JUDGED. ``--append`` calls
        ``gallery.load()`` before the run and ``gallery.save()`` writes
        loaded+new, so judging ``self.embs`` judges a set that never reaches
        the disk. Measured synthetically 2026-09-02: 13 embeddings of him as
        generation 1, then an --append of 13 embeddings of a DIFFERENT
        identity (cross cosine 0.10) passed all four checks and was saved as
        generation 2 under his label -- while the same checks over the 26 that
        landed said [FAIL] cohesion with a worst median of 0.075, and
        ``match()`` then returned ('hunter', 1.0000) for the stranger.
        """
        try:
            return self.gallery.embeddings(self.label)
        except Exception:  # noqa: BLE001 - a gallery that cannot say is not
            # a reason to lose the report; judging this run alone is the
            # old behaviour and is never LOOSER than judging nothing.
            log.debug("faceenrol: gallery could not report its pool",
                      exc_info=True)
            return list(self.embs)

    def pool_takes(self) -> List[Take]:
        """The takes for ``pool()``, index for index. Same fallback: a
        gallery that cannot say gets blanks, which read as "no record"."""
        try:
            return self.gallery.takes(self.label)
        except Exception:  # noqa: BLE001 - see pool()
            log.debug("faceenrol: gallery could not report its takes",
                      exc_info=True)
            return []

    def report(self, identity_min: float = SFACE_COSINE_SAME
               ) -> EnrolmentReport:
        rep = EnrolmentReport(label=self.label, frames=self.frames,
                              detector_ok=True,
                              min_conf=self.limits.min_conf)
        summarise(rep, self.samples, self.embs, self.embed_ms, identity_min,
                  pool=self.pool(), takes=self.pool_takes())
        return rep


# ------------------------------------------------------------- the run loop
def _source_reason(source) -> str:
    """What the frame source says about its own silence, or "".

    Read through a try because ``source`` is duck-typed -- cv2's own
    VideoCapture has no ``reason`` at all, and a property that raises must
    cost the better sentence, never the run.
    """
    try:
        return str(getattr(source, "reason", "") or "").strip()
    except Exception:            # noqa: BLE001 - a duck-typed source
        log.debug("faceenrol: the source could not say why", exc_info=True)
        return ""


# What ``run_enrolment`` says when the frames stop and the source itself
# offers no better account. It is deliberately the LAST resort: it guesses at
# two causes and on 2026-09-03 both guesses were wrong at once -- sensing said
# camera=True and the device was present, merely HELD by the running Jarvis --
# which sent him to check the two things that were already fine. Any source
# that can say something true says it through ``source.reason``.
FRAMES_STOPPED = ("the frame source stopped delivering -- sensing denied the "
                  "camera, or the device went away")


def run_enrolment(session: EnrolmentSession, source, *,
                  plan: Sequence[Station] = DEFAULT_PLAN,
                  say: Callable[[str], None] = print,
                  wait: Optional[Callable[[str], None]] = None,
                  frames_per_station: int = 240,
                  gap_s: float = 0.35,
                  identity_min: float = SFACE_COSINE_SAME,
                  now: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep,
                  on_station: Optional[Callable] = None,
                  on_sample: Optional[Callable] = None,
                  should_stop: Optional[Callable[[], str]] = None) -> Tuple:
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

    THE THREE CALLBACKS ARE THE IN-APP SEAM, and they are additive and
    default-None so that ``scripts/face_enrol.py`` -- which passes none of
    them -- behaves exactly as it did. They exist because the same station
    loop has to drive two very different progress channels: a terminal he
    reads, and a spoken cadence he HEARS with his head turned away from the
    screen (jarvis/enrolrun.py). Routing the spoken version through ``say``
    and matching on the strings was the alternative and it is brittle:
    telling "[lens] Look straight..." from ``Sample.line()`` from "    only 2
    of 3 here" by prefix would break silently the first time one of those
    format strings was edited.

    * ``on_station(station, index, total)`` replaces the three ``say`` lines
      that announce a station. ``index`` is 0-based; ``total`` is len(plan).
    * ``on_sample(sample, station, got, wanted)`` replaces ``say(sample.line())``.
      ``got`` is the count BEFORE this sample is counted, so a caller that
      wants "that is three" adds one for an accepted sample itself.
    * ``should_stop()`` is asked once per frame, at the top of the inner
      loop, and a non-empty return becomes the report's stop ``reason``.
      That is how the window's "Jarvis, stop" reaches a loop that is
      otherwise busy for a minute and a half.
    """
    stations_run = 0
    t_start = now()
    stopped = ""
    total = len(plan)
    for index, station in enumerate(plan):
        if stopped:
            break
        if on_station is None:
            say("")
            say("[%s] %s" % (station.key, station.prompt))
            say("    hold it; %d samples wanted in this position."
                % station.samples)
        else:
            on_station(station, index, total)
        if wait is not None:
            wait("    press Enter when you are in position: ")
        got = 0
        last = 0.0
        for _ in range(int(frames_per_station)):
            if got >= station.samples:
                break
            if should_stop is not None:
                why = should_stop()
                if why:
                    stopped = str(why)
                    break
            ok, frame = source.read()
            if not ok or frame is None:
                # THE SOURCE GETS THE FIRST WORD. It is the only party that
                # knows whether it was denied, unplugged or merely busy, and
                # the hard-coded sentence below has already been wrong about
                # all three at once. See camera.FeedSource.reason.
                stopped = _source_reason(source) or FRAMES_STOPPED
                break
            sample = session.offer(frame, station.key,
                                   note=station.note or station.key)
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
            if on_sample is None:
                say(sample.line())
            else:
                on_sample(sample, station, got, station.samples)
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
def build_models(cfg):
    """``(detector, recogniser, reason)`` -- never raises, never falls back.

    The detector is floored LOW rather than at his ``camera.min_conf``, for
    the same reason scripts/vision_selfcheck.py does it: a face scoring 0.45
    against a 0.6 bar must be reported WITH ITS SCORE, not vanish and read as
    "no face seen". The bar is then applied by the quality gate, which says
    which bar it was.

    THIS LIVES HERE SO BOTH ENROLMENT PATHS LOAD THE SAME PAIR. It was
    written out inside ``scripts/face_enrol.py``, which was fine while that
    script was the only caller; the in-app run (jarvis/enrolrun.py) is a
    second one, and two copies of "which models does enrolment use" is
    exactly how a box ends up enrolling ArcFace vectors through an SFace
    recogniser. The script delegates here and its behaviour is unchanged.

    ``camera`` and ``facedetect`` are imported inside the function to keep
    this module importable with neither of them wired.
    """
    from jarvis import camera as cam          # noqa: PLC0415 - lazy by design
    from jarvis import facedetect             # noqa: PLC0415

    detector, why = cam.detector_from_config(
        cfg, score_threshold=facedetect.PROBE_THRESHOLD)
    if detector is None:
        return None, None, why
    try:
        rec = facedetect.load_recogniser(
            min_conf=float(cfg.get("camera.min_conf", 0.6)),
            model_dir=str(cfg.get("camera.model_dir", "") or "") or None,
            backend=cam.face_backend_from_config(cfg),
            input_size=(int(cfg.get("camera.detect_width", 320)),
                        int(cfg.get("camera.detect_height", 180))))
    except Exception as exc:  # noqa: BLE001 - absence is not a crash
        return detector, None, str(exc)
    return detector, rec, ""


def save_enrolment(gallery, report, *, reason: str,
                   allow_shrink: bool = False, force: bool = False,
                   superseded: Sequence = ()) -> dict:
    """Write this run's pool, but ONLY if the judge passed it. Numbers back.

    THE VERDICT GUARD LIVES HERE RATHER THAN IN THE CALLER, and that is the
    whole point of the function. It used to be a single ``if rep.ok or
    args.force:`` inside ``scripts/face_enrol.py``, which meant the rule
    "an unusable gallery is not saved" was a property of ONE script rather
    than of enrolment -- and the in-app run (jarvis/enrolrun.py) would have
    had to restate it correctly to be safe. A rule that has to be restated
    to hold is a rule that will eventually be restated wrong.

    IT IS DELIBERATELY NOT INSIDE ``FaceGallery.save()``. ``restore()`` and
    ``purge_label()`` legitimately write pools that ``judge_gallery`` never
    ran on, so a guard down there would refuse the two operations whose
    entire job is to put back a pool that already exists.

    ``force`` is the CLI's ``--force`` and nothing else reaches it: the
    in-app path has no way to pass it, which is what makes "the window
    cannot save an unusable gallery" a structural fact rather than a
    promise.

    ``superseded`` is destroyed only AFTER the new generation is safely on
    disk -- ``--reset`` destroys nothing when the run fails a check.
    """
    out = {"saved_generation": 0, "gallery_total": 0, "removed": 0,
           "refused": "", "error": ""}
    if not (report.ok or force):
        out["refused"] = ("the judge refused this pool" if not report.reason
                          else report.reason)
        return out
    try:
        gen = gallery.save(reason=str(reason),
                           allow_shrink=bool(allow_shrink))
    except ValueError as exc:
        # A refusal from the store itself (a shrink, a bad root). The report
        # keeps the FIRST reason it had: the judge's verdict explains more
        # than the store's complaint about the consequence of it.
        report.reason = report.reason or str(exc)
        out["error"] = str(exc)
        return out
    report.saved_generation = int(gen)
    report.gallery_total = int(gallery.total())
    out["saved_generation"] = int(gen)
    out["gallery_total"] = int(report.gallery_total)
    if superseded:
        out["removed"] = int(gallery.drop_generations(superseded))
    return out


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

    TWO WAYS THIS COMMAND USED TO DESTROY THE THING IT PROTECTS.

    1. ``--backup ~/.aiws_trainer/face_gallery`` -- the gallery itself, one
       typo away from the path in every doc -- opened each live generation
       with ``O_TRUNC`` and rewrote it from itself. It reported
       ``{'copied': 1, 'verified': 1, 'ok': True}`` while doing it, and an
       interruption after the truncate left ``gen-00001.npz`` at 0 bytes with
       ``load()`` returning False: the enrolment gone, from the command whose
       job is to keep it. So a destination that IS the gallery, or lives
       inside it, is refused before a byte is written.
    2. The copy was the one non-atomic write in the feature. ``FaceGallery.
       save`` is deliberately tmp + ``os.replace`` and says why; this now
       does the same, so a crash mid-copy costs the tmp and never the copy it
       is replacing.

    The report says what the DESTINATION HOLDS afterwards, not only what was
    copied into it. Those differ after a ``--rollback``: the rolled-back
    generation stays in the backup, it is the newest thing there, and it is
    therefore exactly what a later ``--restore`` would take.
    """
    dest = Path(dest)
    out: dict = {"dest": str(dest), "copied": 0, "verified": 0,
                 "generations": [], "samples": 0, "failed": [],
                 "dest_generations": [], "dest_newest": 0,
                 "dest_newest_samples": 0, "not_in_live": []}
    root = None if gallery.root is None else Path(gallery.root)
    if root is not None:
        try:
            rdest, rroot = dest.resolve(), root.resolve()
        except OSError:                      # a path that cannot be resolved
            rdest, rroot = dest.absolute(), root.absolute()
        if rdest == rroot or rroot in rdest.parents:
            out["failed"].append(
                "%s is the gallery itself (or inside it): a backup into the "
                "gallery is not a backup, and copying a generation over "
                "itself is how the only copy gets truncated. Nothing was "
                "written." % dest)
            out["ok"] = False
            return out
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(dest, 0o700)
    for gen in gallery.generations():
        src = gallery.path_for(gen)
        target = dest / src.name
        tmp = target.with_name(target.name + ".tmp")
        try:
            data = src.read_bytes()
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
            out["copied"] += 1
        except OSError as exc:
            try:
                tmp.unlink()
            except OSError:
                log.debug("could not remove the failed copy %s", tmp.name,
                          exc_info=True)
            out["failed"].append("%s: %s" % (src.name, exc))
            continue
        check = FaceGallery(root=dest, model=gallery.model)
        if check.load(generation=gen):
            out["verified"] += 1
            out["generations"].append(gen)
            out["samples"] += check.total()
        else:
            out["failed"].append("%s: the copy did not parse" % target.name)
    holding = FaceGallery(root=dest, model=gallery.model)
    dest_gens = holding.generations()
    out["dest_generations"] = list(dest_gens)
    out["not_in_live"] = [g for g in dest_gens
                          if g not in set(gallery.generations())]
    if dest_gens:
        out["dest_newest"] = dest_gens[-1]
        newest = FaceGallery(root=dest, model=gallery.model)
        if newest.load(generation=dest_gens[-1]):
            out["dest_newest_samples"] = newest.total()
    out["ok"] = bool(out["verified"] and not out["failed"])
    return out


def restore(gallery: FaceGallery, src: Path, reason: str) -> dict:
    """Load the newest generation from ``src`` and save it as a NEW
    generation of the live gallery.

    Never a file copy over the live store: restoring by overwriting is the
    move that made the 2026-09-02 loss unrecoverable. A restore that turns
    out to be the wrong one is then just another generation to roll back.

    WHAT IT TAKES IS THE NEWEST GENERATION IN ``src``, WHICH IS NOT
    NECESSARILY THE ONE HE MEANT. A backup directory is never reconciled with
    the live gallery, so after a ``--rollback`` the discarded generation is
    still there and is still the newest -- a restore would resurrect exactly
    the enrolment he had just undone. That cannot be guessed at from here, so
    the numbers that decide it are returned and printed: which generation was
    taken, how many samples it holds, and what was live before.
    """
    src = Path(src)
    out: dict = {"src": str(src), "restored": 0, "generation": 0,
                 "samples": 0, "reason": "", "src_generation": 0,
                 "src_samples": 0, "live_before": 0,
                 "live_generation_before": 0}
    out["live_before"] = gallery.total()
    out["live_generation_before"] = gallery.loaded_generation
    # SAME MODEL AS THE LIVE GALLERY, or the restore silently does
    # nothing: a backup written by the other model does not load here,
    # and the honest report is "nothing readable" rather than a merge
    # of two incomparable vector spaces.
    holding = FaceGallery(root=src, model=gallery.model)
    if not holding.load():
        out["reason"] = "nothing readable in %s" % src
        return out
    out["src_generation"] = holding.loaded_generation
    out["src_samples"] = holding.total()
    # A restore REPLACES; it does not merge. Adding a backup's embeddings on
    # top of whatever the live pool happens to be holding produces a pool
    # that is two enrolments at once, whose provenance says "restore" and
    # whose contents are half something else -- which is exactly the kind of
    # store nobody can reason about afterwards. The generations underneath
    # are untouched, so the thing being replaced is still on disk.
    gallery.reset()
    for label in holding.labels():
        # zip, not two loops: a restore that dropped the notes would quietly
        # turn a gallery that knows which poses it covers into one that does
        # not, and nothing about the restored gallery would look wrong.
        for emb, take in zip(holding.embeddings(label),
                             holding.takes(label)):
            gallery.add(label, emb, note=take.note, yaw_deg=take.yaw_deg)
            out["restored"] += 1
    if not out["restored"]:
        out["reason"] = "the newest generation in %s holds no embeddings" % src
        return out
    # allow_shrink: a restore is deliberately a step BACKWARDS, and the
    # shrink guard exists to catch the accidental version of that.
    out["generation"] = gallery.save(reason=reason, allow_shrink=True)
    out["samples"] = gallery.total()
    return out
