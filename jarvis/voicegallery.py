"""The voice gallery: several people's ECAPA embeddings, and the store that
makes multi-speaker voice ID possible at all.

``jarvis/speaker.py`` holds ONE pool with ONE centroid and answers a boolean.
The word "label" does not appear in it. So "say your first and last name and
the voice confirms it" cannot work for anybody but the owner, because there is
nowhere for a second person's vectors to live. This module is that place.

IT IS A NEW FILE IN A NEW DIRECTORY, NOT A NEW FORMAT NUMBER IN
``voiceprint.npz``, and that decision is measured rather than tidy. The old
loader selects keys with ``k.startswith("emb_")``, which matches
``emb_mara_0000`` as happily as ``emb_0000``. Written into that file, several
people pool into one centroid and every one of them is accepted as him --
6 of 6 of a second speaker's synthetic takes cleared the 0.30 bar that way
(tests/test_speaker_format.py). ``speaker.KNOWN_VOICEPRINT_FORMATS`` now
refuses an unrecognised format, but a ROLLBACK is precisely the situation
where that refusal is not present. So the multi-label data never goes near
that filename: new directory, new FORMAT namespace, and ``voiceprint.npz``
left untouched as the thing he reverts to.

THE SHAPE IS ``jarvis/facegallery.py``'s, deliberately and almost line for
line -- generations rather than overwrites, the richest generation never
pruned, the model recorded in the file, a cross-model comparison refused BY
NAME, four write guards, and a delete that reads back before it shreds. That
module is the answer to losing his voiceprint on 2026-09-02 and its reasoning
is written out there; this one inherits it rather than restating it.

WHERE IT DEPARTS FROM THE FACE GALLERY, AND WHY THE NUMBERS FORCED IT.
``FaceGallery.match`` scores against the pool's BEST SAMPLE and argues,
correctly, that a centroid of him in glasses and him without is a face that
does not exist. Voice does not work that way, and copying the face rule here
would have refused him on day one. Measured 2026-09-04 on his own
``voiceprint.npz`` (floats only, no audio):

    sample vs sample, 91 pairs   min 0.283   median 0.485   max 0.735
    sample vs centroid                       0.632 - 0.831
    leave-one-out                            0.570 - 0.798

His own worst PAIR is 0.283 -- BELOW the 0.30 bar that admits him. Nearest-
sample matching would therefore reject one of his own takes against another
of his own takes. Averaging cancels the per-take channel noise and lifts the
same person from ~0.48 to ~0.72, which is the whole reason ``identify``
scores against PER-LABEL CENTROIDS.

WHAT THIS MODULE IS NOT. It is not authentication. ``jarvis/identity.py``
opens by saying a recording defeats the voice check and that the user was told
so; multi-speaker widens the target from one person to N and changes nothing
about that. It is recognition and courtesy.

AND THE HONEST LIMIT, WHICH IS THE FIRST THING ANY READER SHOULD KNOW.
There is not one second of anybody else's voice on this machine. The
"non-match" figures quoted in ``speaker.py`` (-0.03..-0.10) are silence, a
television and room tone -- not a second human. So no between-people bar here
is validated: ``ACCEPT`` is his own measured number and ``MARGIN`` is derived
from splitting HIS pool into two pretend people, which bounds what his own
within-person noise can produce and says nothing about what two real people
produce. Both are labelled where they are defined, and
``scripts/voice_model_compare.py`` refuses to recommend a bar until a second
person is actually enrolled.

Nothing here imports torch, speechbrain or a model. It is arithmetic over
arrays, so it runs in the suite with no microphone and no GPU -- and that is
also why ``identify`` takes an EMBEDDING rather than audio: the trimming and
the encoder live in ``speaker.py``, on the other side of the line that keeps
this file testable.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("voicegallery")

# ITS OWN NAMESPACE, starting again at 1. It is not voiceprint.npz's 2 and
# must never be confused with it: the two files have different key layouts and
# live at different paths precisely so that neither loader can read the other's
# data by accident.
FORMAT = 1

# ------------------------------------------------------------ which model
# The same SpeechBrain encoder speaker.py has always used, named in the file
# for the reason facegallery names its own: an embedding from a different
# encoder is not a worse measurement of the same thing, it is a measurement of
# something else, and the cosine between them means nothing at all. Recorded
# on every write and REQUIRED on every read -- unlike the face gallery there is
# no legacy generation to be generous to, because nothing has ever written
# this store.
ECAPA_MODEL = "ecapa_voxceleb"
MODEL_DIMS = {ECAPA_MODEL: 192}
DEFAULT_MODEL = ECAPA_MODEL


def model_dim(model: str) -> int:
    """How many floats a vector from ``model`` has, or ValueError by name."""
    try:
        return MODEL_DIMS[str(model)]
    except KeyError:
        raise ValueError("unknown voice embedding model %r; known: %s"
                         % (model, ", ".join(sorted(MODEL_DIMS)))) from None


# facegallery's window, for facegallery's reasons: deep enough to undo a
# mistake noticed a few enrolments later, shallow enough that the store stays
# small. The richest generation is never pruned -- five bad writes must not be
# able to evict a good enrolment, which is exactly the state the 2026-09-02
# incident left the voiceprint in.
KEEP_GENERATIONS = 5
MIN_GENERATIONS = 2

_GEN_RE = re.compile(r"^gen-(\d{5})\.npz$")
_TMP_RE = re.compile(r"^gen-(\d{5})\.npz\.tmp$")
_KEY_RE = re.compile(r"^emb_(.+)_(\d{4})$")
# identity.LABEL_RX and facegallery._LABEL_RE, character for character. Three
# stores that disagree about who exists is three stores that silently stop
# naming somebody; this is asserted by a test rather than trusted to a comment.
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")
NOTE_MAX = 120
# The attestation string a guest's enrolment writes. Same shape as the note.
CONSENT_MAX = 120

# ------------------------------------------------------- the collapsed bar
# Median pairwise cosine within one label above which a pool is refused as
# collapsed. A CHOICE INSIDE A MEASURED GAP, and it is labelled as a choice:
# his real 14-take pool sits at 0.485, and the pool the 2026-09-02 accident
# left behind (voiceprint.npz.corrupt-20260902: two rows, every element
# 0.07216878) sits at 1.000. 0.90 is between them with a wide berth on both
# sides. It is NOT a quality bar and it does not certify an enrolment.
#
# SAY WHAT IT DOES NOT CATCH, because the first draft of this comment claimed
# more than the numbers support. Measured 2026-09-04: a pool of twelve
# zero-mean random vectors has a median pairwise cosine of -0.035..0.015 --
# nowhere near 0.90 -- so this guard would wave rubbish through. It catches
# COLLAPSE (the same vector saved N times), which is the shape that actually
# destroyed the voiceprint. Rubbish is caught upstream, by the trimming, the
# VAD and the recorder, and nothing here may be relaxed on the strength of it.
COLLAPSED_MEDIAN_COSINE = 0.90

# --------------------------------------------------------------- the bars
# THE ACCEPT BAR IS NOT RE-DERIVED HERE AND IS NOT RAISED HERE. 0.30 is
# speaker.DEFAULT_THRESHOLD and it is measured: 0.40 sat inside his own
# genuine band of 0.377-0.397 and a tuning run put it at a 40% false-reject
# rate against 0%/0% at 0.30. The live value still comes from
# voice_settings.json's speaker_threshold; this is what a fresh box gets.
# Duplicated rather than imported because speaker.py imports THIS module;
# tests/test_voicegallery.py asserts the two numbers stay equal.
ACCEPT_DEFAULT = 0.30

# THE MARGIN. Both bars must be cleared before anybody is named.
#
# DERIVED, NOT INTUITED, AND STILL NOT A DISCRIMINATION BAR. There is no
# second person's voice on this box, so a real between-people distribution
# cannot be measured. What CAN be measured is the other side: split his own 14
# takes into two pretend people and ask this exact centroid rule to tell them
# apart. Over 2800 trials (2026-09-04) the same-person margin runs
#
#     p50 0.044   p90 0.117   p95 0.141   p99 0.194   max 0.257
#
# and an independent 1400-trial run gave p95 0.140, p99 0.193, max 0.256.
# 0.20 sits just above the p99 of both and below both maxima: 0.75% of
# same-person trials still exceed it.
#
# So this is a FLOOR ON MEANING -- the smallest gap a verdict must clear
# before it is saying something his own within-person noise could not have
# produced by itself. It is NOT evidence that two real people are separable by
# 0.20, and it must never be described as one. PROVISIONAL until a second
# person enrols and scripts/voice_model_compare.py can measure the real thing.
MARGIN = 0.20
MARGIN_IS_PROVISIONAL = True

# A label with fewer than this many takes SCORES AND LOGS BUT NEVER NAMES.
# Measured 2026-09-04 on his pool: the cosine between a centroid built from k
# random takes and the converged 14-take centroid runs k=2 0.841, k=5 0.943,
# k=6 0.956, k=8 0.975, k=10 0.987. Below 8 a centroid is still moving, and a
# moving centroid produces a margin that means less than the margin bar
# assumes. Past 8 the curve is flat enough that more takes buy little.
MIN_TAKES_TO_NAME = 8

# speaker.ABSTAIN_SECONDS, and it is load-bearing rather than shared for
# tidiness: below 1.5 s of trimmed speech verify() does not score at all and
# FAILS OPEN, because every "Yes." he says arrives that short and a score from
# that little speech is a coin flip. identify() abstains at the same point and
# names NOBODY; the OWNER FALLBACK for an abstention lives in gate._voice_leg,
# where the owner's label is actually known. Asserted equal by a test.
ABSTAIN_SECONDS = 1.5

# PASSIVE LEARNING IS OFF, AND THE ZERO IS A MEASUREMENT RATHER THAN A MOOD.
#
# The plan for this store was a frozen reference centroid and a cap of 20.
# Freezing is right and it is implemented -- gating a new sample against the
# centroid it is about to move bounds one STEP and not the WALK, so an
# attacker bootstraps a little further on every accepted sample. But freezing
# turns out to be necessary and NOT sufficient, and the difference was
# measured rather than assumed (2026-09-04, synthetic vectors at his pool's
# spread; tests/test_voice_passive.py reproduces every row):
#
#   worst-case attacker, every sample at the bar, all in one direction,
#   against a 14-take enrolment
#
#     rule                         after 20   cos(new, original)   intruder
#     gate on the LIVE centroid                     0.657            0.843
#     gate on the FROZEN centroid                   0.724            0.790
#     frozen, bar raised to 0.55                    0.783            0.731
#     frozen, bar raised to 0.80                    0.908            0.549
#
# The intruder starts at 0.149 and the accept bar is 0.30, so EVERY row above
# ends with a stranger comfortably inside. No bar in a plausible range fixes
# it, because the problem is not the bar: twenty new samples against fourteen
# originals is a 59% swing in the mean whatever each one scores. Only the
# COUNT bounds it --
#
#     frozen, bar at the label's own leave-one-out floor
#       cap  1 -> cos 0.997, intruder 0.220     (a stranger stays out)
#       cap  2 -> cos 0.991, intruder 0.281     (only just)
#       cap  3 -> cos 0.982, intruder 0.334     (a stranger is now inside)
#
# -- and two samples of adaptation is not worth a door. There is a second,
# worse problem underneath: ``voiceprint.npz`` cannot tell an enrolment take
# from a passively learned one, so "the frozen enrolment centroid" is only
# frozen until the next restart, after which the walk resumes from wherever it
# got to. This store CAN tell them apart (``src_`` = "passive"), which is why
# the arithmetic lives here -- but the weighting is hard toward false reject,
# a false accept hands somebody else his assistant with owner scope, and with
# N people enrolled this is N doors instead of one.
#
# So the default is zero and ``passive_ok`` refuses everything. A caller who
# means it passes ``max_passive`` explicitly and gets the frozen reference,
# the per-person genuine floor, the margin-won label and the cap.
MAX_PASSIVE = 0


@dataclass(frozen=True)
class Take:
    """What one enrolment take was, beyond its 192 floats.

    Every field is OPTIONAL AT THE SAME ``FORMAT``. That is the same argument
    facegallery makes for ``note_``/``yaw_``: a take with no note predates
    notes, it is not an error, and bumping the format for a cosmetic field
    would make an existing enrolment unreadable by the build that added the
    field -- the exact class of loss this store exists to prevent.

    ``len_s`` and ``rms`` are None rather than 0.0 when nothing was recorded,
    for facegallery's ``yaw_deg`` reason: a 0.0 default would report a silent
    take rather than an unmeasured one, turning "this pool cannot say what it
    holds" into a confident and wrong answer.
    """

    len_s: Optional[float] = None      # trimmed speech seconds
    rms: Optional[float] = None        # the take's own loudness
    note: str = ""                     # the enrolment prompt line, verbatim
    at: str = ""                       # when it was captured
    src: str = ""                      # "enrol" | "legacy" | "passive"

    @property
    def recorded(self) -> bool:
        return bool(self.note or self.at or self.src) or \
            self.len_s is not None or self.rms is not None

    def as_dict(self) -> dict:
        return {"len_s": self.len_s, "rms": self.rms, "note": self.note,
                "at": self.at, "src": self.src}


@dataclass(frozen=True)
class VoiceVerdict:
    """What the voice leg has to say. ``who == ""`` IS THE DEFAULT RETURN.

    UNKNOWN is reachable by construction rather than by an exception path:
    every refusal below builds this object with an empty ``who`` and a ``why``
    that names the number it failed on. Nobody is named unless BOTH bars were
    cleared and the label had enough takes for its centroid to have settled.

    ``margin`` is None when only one label is enrolled -- "the bar did not
    apply", which is a different fact from "the bar was cleared by 0.0". The
    same three-valued habit ``facegallery.Take.yaw_deg`` and
    ``jarvis/roomsensor.py`` write down: absent is not zero.

    ``provisional`` carries the label that WOULD have won if it had enough
    takes. It is there so the caller can say "I think that's Mara, but I've
    only heard her a few times" instead of silently saying nothing -- and it
    is never ``who``, so it can never grant scope.
    """

    who: str = ""
    score: float = 0.0
    second: str = ""
    second_score: float = 0.0
    margin: Optional[float] = None
    speech_s: float = 0.0
    why: str = ""
    provisional: str = ""
    abstained: bool = False
    scores: Tuple[Tuple[str, float], ...] = ()

    def as_dict(self) -> dict:
        return {"who": self.who, "score": self.score, "second": self.second,
                "second_score": self.second_score, "margin": self.margin,
                "speech_s": self.speech_s, "why": self.why,
                "provisional": self.provisional, "abstained": self.abstained}


def label_ok(label) -> bool:
    """Is this a name ``add()`` will accept? Public so a script can refuse a
    bad ``--label`` at the ARGUMENT, before eight takes at the microphone."""
    return bool(_LABEL_RE.match(str(label or "")))


def clean_text(text, cap=NOTE_MAX) -> str:
    """One printable line, capped. facegallery.clean_note's rule, and its
    reasoning: this lands in an npz key's value, in a report he pastes
    somewhere, and in a log line, so a newline or a control character would
    break one of the three -- and ``str()`` of an array is a string, so a note
    can never smuggle numbers past the checker under a cosmetic key."""
    s = "" if text is None else str(text)
    s = "".join(c if (c.isprintable() and c != "\x00") else " " for c in s)
    return " ".join(s.split())[:int(cap)]


def clean_float(value) -> Optional[float]:
    """A finite number, or None meaning "not recorded". NaN would compare
    False against every bound while still counting as present."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def cosine(a, b) -> float:
    """Cosine similarity; 0.0 rather than NaN for a zero vector.

    The same normalise-at-compare-time arithmetic ``speaker._cosine_similarity``
    does today. ECAPA vectors are NOT unit length (his run 242-355), so this
    must never be shortened to a dot product."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or not np.isfinite(na) or not np.isfinite(nb):
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def centroid(vectors) -> Optional[np.ndarray]:
    """The RAW mean of a label's vectors, normalised only at compare time --
    exactly what ``speaker._recompute_centroid`` does, so a migrated pool
    scores identically here and there."""
    rows = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if not rows:
        return None
    return np.mean(rows, axis=0)


def median_pairwise(vectors) -> Optional[float]:
    """The pool's own cohesion, or None below two samples."""
    rows = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if len(rows) < 2:
        return None
    vals = [cosine(rows[i], rows[j])
            for i in range(len(rows)) for j in range(i + 1, len(rows))]
    return float(np.median(vals))


def degenerate_reason(vec, model: str = DEFAULT_MODEL) -> str:
    """"" if this could be an ECAPA embedding, else why not.

    facegallery.degenerate_reason's four shapes, at 192 floats: the wrong
    dimension (another model's output), a non-finite element, an all-zero
    vector, and a CONSTANT one -- which is what the fixture that destroyed his
    voiceprint contained (every element 0.07216878) and which no real
    embedding is."""
    dim = model_dim(model)
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size != dim:
        return "wrong dimension: %d, expected %d for %s" % (arr.size, dim, model)
    if not np.all(np.isfinite(arr)):
        return "not finite"
    if float(np.linalg.norm(arr)) == 0.0:
        return "all zero"
    if float(np.std(arr)) < 1e-6:
        return ("constant vector (std %.2e) -- this is a fixture, not a voice"
                % float(np.std(arr)))
    return ""


def _shred(path: Path) -> None:
    """Overwrite a file's bytes, then unlink it. facegallery._shred, and the
    same honest limit applies: this makes the vectors unreachable THROUGH THE
    FILESYSTEM. On an SSD whose controller remaps rather than rewrites it is
    not a device-level erase, and any command built on it may only claim the
    part that is true."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > 0:
        try:
            fd = os.open(path, os.O_WRONLY)
            try:
                chunk = b"\0" * min(size, 1 << 20)
                left = size
                while left > 0:
                    left -= os.write(fd, chunk[:min(left, len(chunk))])
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            log.warning("could not overwrite %s before deleting it",
                        path.name, exc_info=True)
    path.unlink()


def _take_for(data, have, label: str, idx: str) -> Take:
    """One stored take's metadata, and never an exception.

    Contained to ONE take on purpose: a malformed cosmetic key must cost its
    own note, never the generation. facegallery learned this the hard way --
    one bad string raising out of ``_read`` turns into "this generation is
    unreadable", which is thirteen embeddings for a typo."""
    fields = {}
    for name, key, conv in (
            ("len_s", "len_%s_%s" % (label, idx), clean_float),
            ("rms", "rms_%s_%s" % (label, idx), clean_float),
            ("note", "note_%s_%s" % (label, idx), clean_text),
            ("at", "at_%s_%s" % (label, idx), clean_text),
            ("src", "src_%s_%s" % (label, idx), clean_text)):
        if key not in have:
            continue
        try:
            fields[name] = conv(data[key][0])
        except Exception:  # noqa: BLE001 - a bad field costs the field
            log.warning("voice gallery: unreadable %s; the take keeps its "
                        "embedding and loses that field", key, exc_info=True)
    fields["note"] = fields.get("note") or ""
    fields["at"] = fields.get("at") or ""
    fields["src"] = fields.get("src") or ""
    return Take(**fields)


class VoiceGallery:
    """Enrolled voices, held in memory, persisted as numbered generations.

    ``root=None`` is a live, unsaveable gallery -- the same escape hatch
    facegallery offers, and what the arithmetic tests use.
    """

    def __init__(self, root: Optional[Path] = None,
                 model: str = DEFAULT_MODEL):
        self.root: Optional[Path] = None if root is None else Path(root)
        self.model = str(model)
        model_dim(self.model)          # raise now, not on somebody's enrolment
        self.foreign_generations: Dict[int, str] = {}
        self._pool: Dict[str, List[np.ndarray]] = {}
        self._takes: Dict[str, List[Take]] = {}
        self._consent: Dict[str, str] = {}
        self.loaded_generation = 0
        self._loaded_n = 0
        self._provenance: dict = {}

    # ------------------------------------------------------------ in memory
    def labels(self) -> Tuple[str, ...]:
        return tuple(sorted(k for k, v in self._pool.items() if v))

    def count(self, label: str) -> int:
        return len(self._pool.get(label, ()))

    def total(self) -> int:
        return sum(len(v) for v in self._pool.values())

    def embeddings(self, label: str) -> List[np.ndarray]:
        return list(self._pool.get(label, ()))

    def consent(self, label: str) -> str:
        return self._consent.get(str(label), "")

    def takes(self, label: str) -> List[Take]:
        """Index for index with ``embeddings(label)``, padded rather than
        short: every caller zips the two, and a short list would attach one
        take's note to another take's voice."""
        n = len(self._pool.get(label, ()))
        got = list(self._takes.get(label, ()))
        if len(got) < n:
            got = got + [Take()] * (n - len(got))
        return got[:n]

    def centroid(self, label: str) -> Optional[np.ndarray]:
        return centroid(self._pool.get(label, ()))

    def centroids(self) -> Dict[str, np.ndarray]:
        """Every enrolled label's centroid. What ``identify`` scores against
        and what ``speaker.score`` takes its maximum over."""
        out = {}
        for label, pool in self._pool.items():
            if pool:
                out[label] = centroid(pool)
        return out

    def provisional(self, label: str) -> bool:
        """Too few takes for this label's centroid to have settled. It still
        scores and still logs; it may never NAME anybody."""
        return 0 < self.count(label) < MIN_TAKES_TO_NAME

    def reset(self) -> None:
        """Empty the in-memory pool. Touches no disk, so "start the enrolment
        again" is a safe thing to do."""
        self._pool = {}
        self._takes = {}
        self._consent = {}

    def add(self, label: str, vec, *, len_s=None, rms=None, note="",
            at="", src="") -> None:
        """Add one embedding and what it was, or raise ValueError saying why
        not. The take is appended only after every refusal has passed, so the
        two lists cannot come apart on a rejected sample."""
        if not _LABEL_RE.match(str(label or "")):
            raise ValueError("bad label %r: lowercase letters, digits, - and _"
                             % (label,))
        why = degenerate_reason(vec, self.model)
        if why:
            raise ValueError("refusing a degenerate embedding: %s" % why)
        arr = np.asarray(vec, dtype=np.float32).ravel().copy()
        self._pool.setdefault(label, []).append(arr)
        self._takes.setdefault(label, []).append(
            Take(len_s=clean_float(len_s), rms=clean_float(rms),
                 note=clean_text(note), at=clean_text(at),
                 src=clean_text(src)))

    def set_consent(self, label: str, attestation: str) -> None:
        """Record HOW consent was taken for this label.

        Stored beside the vectors rather than in a separate ledger because a
        pool and its attestation must not be able to come apart: a generation
        that holds somebody's voice and cannot say why it is allowed to is a
        generation nobody can act on."""
        if not _LABEL_RE.match(str(label or "")):
            raise ValueError("bad label %r" % (label,))
        self._consent[str(label)] = clean_text(attestation, CONSENT_MAX)

    def forget(self, label: str) -> int:
        """Drop one person from the IN-MEMORY pool; ``save()`` commits it.

        NOT A DELETE, exactly as facegallery.forget is not: the generations on
        disk still hold them and one ``rollback()`` brings them back.
        Withdrawn consent is ``purge_label()``."""
        gone = len(self._pool.pop(str(label), ()))
        self._takes.pop(str(label), None)
        self._consent.pop(str(label), None)
        return gone

    # ---------------------------------------------------------- the verdict
    def identify(self, vec, speech_s: float,
                 threshold: Optional[float] = None) -> VoiceVerdict:
        """Who this embedding is, or UNKNOWN. NOBODY IS NAMED BY DEFAULT.

        Takes an EMBEDDING and the seconds of TRIMMED SPEECH it came from --
        not audio. The trim and the encoder live in ``speaker.py``; keeping
        them there is what lets this whole decision be tested with no
        microphone and no GPU, which on this box is the only way it can be
        tested at all.

        SCORE IS COSINE AGAINST THE PER-LABEL CENTROID, and this is the one
        place the design deliberately departs from ``facegallery.match``.
        That method scores against the pool's BEST SAMPLE and argues, rightly
        for faces, that a centroid of him in glasses and him without is a face
        that does not exist. Voice is measurably the other way round. On his
        own pool (2026-09-04): sample-to-sample cosine bottoms at 0.283 --
        BELOW the 0.30 bar that admits him -- while sample-to-centroid runs
        0.632-0.831 and leave-one-out 0.570-0.798. Nearest-sample matching
        would reject one of his takes against another of his takes on day one.

        TWO BARS, BOTH REQUIRED:

        1. ACCEPT -- top1 >= ``threshold``, default ``ACCEPT_DEFAULT`` (0.30,
           measured; the live value comes from voice_settings.json). Nothing
           here re-derives it and nothing here raises it.
        2. MARGIN -- (top1 - top2) >= ``MARGIN`` (0.20), applied ONLY when a
           second label is enrolled. Derived from splitting his own pool into
           two pretend people, so it bounds what his own within-person noise
           can produce; it is NOT a validated discrimination bar and the
           constant's comment says so at length.

        Fail either and ``who`` is "" with a ``why`` naming the number. That
        is UNKNOWN, and it is the sentence "I can hear someone I know, but I
        can't tell which of you" rather than a coin flip.

        ABSTAIN, UNCHANGED AND LOAD-BEARING. Below ``ABSTAIN_SECONDS`` of
        trimmed speech nothing is scored at all: a score from that little
        speech is a coin flip (FRR@0.30 measured at 30% on 1.0 s against 0% at
        3.0 s), and every "Yes." he says arrives that short. The verdict comes
        back with ``abstained`` True and ``who`` "" -- and THE OWNER FALLBACK
        FOR AN ABSTENTION LIVES IN ``gate._voice_leg``, where the owner's
        label is actually known, because this module is arithmetic and has no
        business deciding who owns the machine. An abstention is a fail-open,
        not a recognition, and must never be narrated as one.

        PROVISIONAL. A label with fewer than ``MIN_TAKES_TO_NAME`` takes
        scores and logs but never names -- its centroid has not settled (cos
        to the converged position measures 0.841 at k=2 against 0.975 at k=8).
        """
        speech_s = float(speech_s or 0.0)
        bar = ACCEPT_DEFAULT if threshold is None else float(threshold)
        if speech_s < ABSTAIN_SECONDS:
            return VoiceVerdict(speech_s=speech_s, abstained=True,
                                why="abstain: %.2fs of speech is under %.2fs"
                                    % (speech_s, ABSTAIN_SECONDS))
        cents = self.centroids()
        if not cents:
            return VoiceVerdict(speech_s=speech_s, why="nobody is enrolled")
        why_bad = degenerate_reason(vec, self.model)
        if why_bad:
            return VoiceVerdict(speech_s=speech_s,
                                why="no usable embedding: %s" % why_bad)
        ranked = sorted(((label, cosine(vec, c)) for label, c in cents.items()),
                        key=lambda kv: kv[1], reverse=True)
        top, top_s = ranked[0]
        second, second_s = ranked[1] if len(ranked) > 1 else ("", 0.0)
        margin = (top_s - second_s) if second else None
        common = {"score": float(top_s), "second": second,
                  "second_score": float(second_s), "margin": margin,
                  "speech_s": speech_s,
                  "scores": tuple((k, float(v)) for k, v in ranked)}

        if top_s < bar:
            # Indistinguishable from an unrelated voice, on purpose: this is
            # the branch the ordinary UNKNOWN line answers.
            return VoiceVerdict(why="best %.3f below %.2f" % (top_s, bar),
                                **common)
        if margin is not None and margin < MARGIN:
            return VoiceVerdict(
                why="margin %.3f below %.2f (%s %.3f, %s %.3f)"
                    % (margin, MARGIN, top, top_s, second, second_s),
                **common)
        if self.provisional(top):
            return VoiceVerdict(
                provisional=top,
                why="%s has only %d take(s); %d before a name"
                    % (top, self.count(top), MIN_TAKES_TO_NAME),
                **common)
        return VoiceVerdict(who=top, why="%s at %.3f" % (top, top_s), **common)

    def genuine_floor(self, label: str) -> Optional[float]:
        """The BOTTOM of this person's own measured band: the smallest
        leave-one-out cosine across their enrolment takes.

        A per-person, measured bar rather than a global guess. On his real
        pool it is 0.570 (2026-09-04, 14 takes, LOO range 0.570-0.798) against
        an accept bar of 0.30 -- so a sample only just past the accept bar is
        nowhere near as characteristic of him as his own worst enrolment take,
        and has no business being added to the pool that defines him.

        None below two enrolment takes: there is no leave-one-out to take.
        """
        pool = self._enrolment_vectors(str(label))
        if len(pool) < 2:
            return None
        vals = []
        for i in range(len(pool)):
            rest = centroid([e for j, e in enumerate(pool) if j != i])
            vals.append(cosine(pool[i], rest))
        return float(min(vals))

    def passive_ok(self, label: str, vec, verdict: VoiceVerdict,
                   frozen: Optional[np.ndarray] = None,
                   max_passive: Optional[int] = None) -> Tuple[bool, str]:
        """May this accepted sample join ``label``'s pool? ``(ok, why not)``.

        OFF BY DEFAULT -- ``MAX_PASSIVE`` is 0, and that constant's comment
        carries the measurement that put it there. This method is the
        arithmetic a caller gets if it deliberately turns passive learning on
        by passing ``max_passive``, and every condition below is a measured
        direction rather than a preference:

        1. Scored against a FROZEN reference centroid -- the enrolment's --
           never the live one. Gating a sample against the centroid it is
           about to move bounds one STEP and not the WALK: measured, twenty
           at-threshold accepts under the live rule rotate the centroid to cos
           0.657 of where it started and lift an intruder from 0.149 to 0.843.
           Freezing alone gets that to 0.724 / 0.790, which is better and is
           still a stranger inside the door -- hence 2 and 4.
        2. Only for a label the verdict NAMED, which means it cleared BOTH the
           accept bar and the margin. A near miss teaches nobody.
        3. Never for a provisional label: a centroid that has not settled must
           not be moved by evidence it chose for itself.
        4. At least this person's own GENUINE FLOOR, not the accept bar. The
           accept bar is where recognition starts; the floor is where THIS
           person's own worst enrolment take sits. Adding something weaker
           than his own weakest take is how the pool stops being about him.
        5. At most ``max_passive`` passive takes, ever, and the count is read
           from the stored ``src_`` keys rather than remembered -- which is the
           thing ``voiceprint.npz`` cannot do, and the reason the walk resumes
           there after every restart.
        """
        label = str(label)
        cap = MAX_PASSIVE if max_passive is None else int(max_passive)
        if cap <= 0:
            return False, ("passive learning is off (MAX_PASSIVE is 0): no bar "
                           "bounds the drift, only the count does, and two "
                           "samples of adaptation is not worth the door")
        if verdict is None or verdict.who != label:
            return False, "the verdict did not name %s" % label
        if verdict.abstained:
            return False, "an abstention is not a recognition"
        if self.provisional(label):
            return False, ("%s is provisional (%d of %d takes); a centroid "
                           "that has not settled may not teach itself"
                           % (label, self.count(label), MIN_TAKES_TO_NAME))
        passive = sum(1 for t in self.takes(label) if t.src == "passive")
        if passive >= cap:
            return False, ("%s already holds %d passive sample(s), the cap"
                           % (label, passive))
        why = degenerate_reason(vec, self.model)
        if why:
            return False, why
        ref = self.enrolment_centroid(label) if frozen is None else frozen
        if ref is None:
            return False, "%s has no enrolment centroid to compare against" % label
        bar = self.genuine_floor(label)
        if bar is None:
            return False, "%s has too few takes to have a genuine floor" % label
        score = cosine(vec, ref)
        if score < bar:
            return False, ("%.3f against the frozen enrolment centroid, below "
                           "%s's own genuine floor of %.3f" % (score, label, bar))
        return True, ""

    def _enrolment_vectors(self, label: str) -> List[np.ndarray]:
        """This label's NON-PASSIVE takes, or the whole pool when nothing is
        marked -- which is what a migrated legacy pool looks like, and every
        one of those takes did come from a deliberate enrolment."""
        pool = self._pool.get(str(label), [])
        tks = self.takes(str(label))
        kept = [e for e, t in zip(pool, tks) if t.src != "passive"]
        return kept or list(pool)

    def enrolment_centroid(self, label: str) -> Optional[np.ndarray]:
        """The centroid of this label's NON-PASSIVE takes -- the frozen
        reference ``passive_ok`` measures against.

        Falls back to the whole pool only when nothing is marked, which is
        what a migrated legacy pool looks like: every one of those takes came
        from a deliberate enrolment, so the two answers are the same."""
        return centroid(self._enrolment_vectors(str(label)))

    # ------------------------------------------------------------- on disk
    def path_for(self, generation: int) -> Path:
        if self.root is None:
            raise ValueError("this gallery has no root; it cannot be saved")
        return self.root / ("gen-%05d.npz" % generation)

    def generations(self) -> List[int]:
        if self.root is None or not self.root.is_dir():
            return []
        out = []
        for p in self.root.iterdir():
            m = _GEN_RE.match(p.name)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    def _tmp_paths(self) -> List[Path]:
        if self.root is None or not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if _TMP_RE.match(p.name))

    def leftovers(self) -> List[str]:
        """Names of any crashed-save ``.tmp``. Each holds a WHOLE pool under a
        name ``generations()`` is blind to, so one surviving a delete is a
        failed delete, not an untidy directory."""
        return sorted(p.name for p in self._tmp_paths())

    def load(self, generation: Optional[int] = None) -> bool:
        """Load one generation, defaulting to the newest that PARSES.

        Falling back down the stack is the whole reason the stack exists: a
        truncated newest file costs the last enrolment, not the enrolment.
        """
        wanted = [generation] if generation else list(reversed(self.generations()))
        self.foreign_generations = {}
        for gen in wanted:
            try:
                pool, takes, consent, prov = self._read(self.path_for(gen))
            except Exception:  # noqa: BLE001 - unreadable means try the one before
                log.warning("voice gallery generation %d unreadable; falling "
                            "back to the one before", gen, exc_info=True)
                continue
            wrote = str(prov.get("model") or "")
            if wrote != self.model:
                # REFUSED BY NAME, not scaled and not truncated. The file stays
                # exactly where it is -- it is a previous enrolment.
                self.foreign_generations[gen] = wrote or "unnamed"
                log.warning(
                    "voice gallery generation %d was written by %r and this "
                    "gallery is %r. NOT comparing across models: the cosine "
                    "between them measures nothing. Those %d sample(s) stay "
                    "on disk untouched.", gen, wrote or "an unnamed model",
                    self.model, int(prov.get("n") or 0))
                continue
            self._pool = pool
            self._takes = takes
            self._consent = consent
            self.loaded_generation = gen
            self._loaded_n = sum(len(v) for v in pool.values())
            self._provenance = prov
            log.info("voice gallery loaded: generation %d, %d samples over %d "
                     "label(s) (%s), model %s", gen, self._loaded_n, len(pool),
                     ", ".join(sorted(pool)) or "-", self.model)
            return True
        return False

    def _read(self, path: Path):
        data = np.load(path)
        names = list(data.files)
        fmt = int(data["_format"][0]) if "_format" in names else 0
        if fmt != FORMAT:
            raise ValueError("voice gallery format %d, this build reads %d"
                             % (fmt, FORMAT))
        # THE MODEL IS REQUIRED, unlike the face gallery's optional key. There
        # is no legacy generation here to be generous to: nothing has ever
        # written this store, so a file with no ``_model`` was not written by
        # this code and its vectors cannot be vouched for.
        if "_model" not in names:
            raise ValueError("voice gallery %s names no model; refusing to "
                             "guess what encoder produced its vectors"
                             % path.name)
        wrote = str(data["_model"][0])
        try:
            dim = model_dim(wrote)
        except ValueError:
            # Written by a build that knows an encoder this one does not. Not
            # corrupt -- REPORTED, so a caller can say the true sentence, and
            # left alone, because "newer" and "truncated" want opposite
            # responses and destroying the wrong one is unrecoverable.
            log.warning("voice gallery %s was written by unknown model %r; "
                        "loading no vectors from it and leaving it alone",
                        path.name, wrote)
            return {}, {}, {}, {"format": fmt, "created_ns": 0, "reason": "",
                               "n": 0, "recorded": 0, "model": wrote}
        if "_dim" in names:
            try:
                stated = int(data["_dim"][0])
            except Exception:  # noqa: BLE001 - an unreadable width is a refusal
                raise ValueError("voice gallery %s has an unreadable _dim"
                                 % path.name) from None
            if stated != dim:
                raise ValueError(
                    "voice gallery %s says its vectors are %d wide and %s is "
                    "%d wide" % (path.name, stated, wrote, dim))
        pool: Dict[str, List[np.ndarray]] = {}
        takes: Dict[str, List[Take]] = {}
        have = set(names)
        for key in sorted(names):
            m = _KEY_RE.match(key)
            if not m:
                continue
            arr = np.asarray(data[key], dtype=np.float32).ravel()
            why = degenerate_reason(arr, wrote)
            if why:
                # A stored vector that cannot be a voice would drag every
                # future match toward itself. Its take goes with it: the lists
                # are paired by position, so keeping a dropped vector's note
                # shifts every later note onto the wrong person.
                log.warning("voice gallery: dropping %s (%s)", key, why)
                continue
            label, idx = m.group(1), m.group(2)
            pool.setdefault(label, []).append(arr)
            takes.setdefault(label, []).append(_take_for(data, have, label, idx))
        consent: Dict[str, str] = {}
        for label in pool:
            key = "consent_%s" % label
            if key in have:
                try:
                    consent[label] = clean_text(data[key][0], CONSENT_MAX)
                except Exception:  # noqa: BLE001 - see _take_for
                    log.warning("voice gallery: unreadable %s", key,
                                exc_info=True)
        prov = {
            "format": fmt,
            "created_ns": (int(data["_created_ns"][0])
                           if "_created_ns" in have else 0),
            "reason": str(data["_reason"][0]) if "_reason" in have else "",
            "n": sum(len(v) for v in pool.values()),
            "recorded": sum(1 for ts in takes.values() for t in ts if t.recorded),
            "model": wrote,
        }
        return pool, takes, consent, prov

    # ------------------------------------------------- whose data is in here
    def _inventory(self, path: Path):
        """``({label: samples}, model)`` by RAW key read, or ``(None, "")``.

        A DIFFERENT QUESTION FROM ``_read``'s, and facegallery's hardest-won
        lesson. ``_read`` yields VECTORS, and every filter it applies -- the
        model check, the width check, the degenerate check -- is a way for its
        answer to come back "nobody is in here" over a file with somebody's
        name in eight of its keys. A delete that trusted ``_read`` left her on
        the disk and reported success. This reads the NAMES."""
        try:
            with np.load(path) as data:
                names = list(data.files)
                if "_format" not in names:
                    return None, ""
                if int(np.asarray(data["_format"]).ravel()[0]) != FORMAT:
                    return None, ""
                wrote = ""
                if "_model" in names:
                    try:
                        wrote = str(np.asarray(data["_model"]).ravel()[0])
                    except Exception:  # noqa: BLE001
                        wrote = "unknown"
        except Exception:  # noqa: BLE001 - unreadable is an ANSWER here
            return None, ""
        counts: Dict[str, int] = {}
        for key in names:
            m = _KEY_RE.match(key)
            if m:
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        return counts, wrote

    def disk_labels(self) -> Tuple[str, ...]:
        """Every label with embeddings ON THIS DISK, whatever model wrote it.
        ``labels()`` is only what this object managed to LOAD."""
        found = set()
        for gen in self.generations():
            counts, _wrote = self._inventory(self.path_for(gen))
            for label, n in (counts or {}).items():
                if n:
                    found.add(label)
        return tuple(sorted(found))

    # ------------------------------------------------------------ the write
    def save(self, reason: str, allow_shrink: bool = False,
             prune: bool = True) -> int:
        """Write the pool as the NEXT generation; return its number.

        ``reason`` is provenance, not decoration: when a store turns out to be
        wrong the only question that matters is what wrote it, and on
        2026-09-02 nothing on disk could answer that.

        ``prune=False`` is for ``purge_label``, whose contract is that no file
        dies unproven -- pruning inside a deliberate delete destroyed a
        bystander's only generation in the face gallery before the delete loop
        had considered a single file.
        """
        if self.root is None:
            raise ValueError("this gallery has no root; it cannot be saved")
        n = self.total()
        if n == 0:
            raise ValueError("refusing to save an empty gallery")
        self._check_not_collapsed()
        baseline = max(self._loaded_n, self._on_disk_n())
        if not allow_shrink and baseline and n < baseline:
            raise ValueError(
                "refusing to shrink the voice gallery from %d samples to %d; "
                "pass allow_shrink=True if you mean it (this guard exists "
                "because the voiceprint went 6 -> 2 unnoticed on 2026-09-02)"
                % (baseline, n))

        # 0o700 on the mkdir ITSELF, not only on a chmod after it: under his
        # umask a plain mkdir creates 0775 and stays world-readable until the
        # chmod lands. These are biometric measurements of several people.
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        gen = (self.generations() or [0])[-1] + 1
        arrays: Dict[str, np.ndarray] = {}
        for label, pool in self._pool.items():
            tks = self.takes(label)
            for i, emb in enumerate(pool):
                arrays["emb_%s_%04d" % (label, i)] = emb
                # WRITTEN ONLY WHEN THERE IS SOMETHING TO WRITE, so a pool
                # with no metadata produces the same file it would have before
                # these keys existed -- which is what keeps a take with no note
                # a take rather than an error.
                t = tks[i]
                if t.len_s is not None:
                    arrays["len_%s_%04d" % (label, i)] = \
                        np.array([float(t.len_s)])
                if t.rms is not None:
                    arrays["rms_%s_%04d" % (label, i)] = \
                        np.array([float(t.rms)])
                if t.note:
                    arrays["note_%s_%04d" % (label, i)] = np.array([t.note])
                if t.at:
                    arrays["at_%s_%04d" % (label, i)] = np.array([t.at])
                if t.src:
                    arrays["src_%s_%04d" % (label, i)] = np.array([t.src])
            if self._consent.get(label):
                arrays["consent_%s" % label] = np.array([self._consent[label]])
        arrays["_format"] = np.array([FORMAT])
        arrays["_model"] = np.array([str(self.model)])
        arrays["_dim"] = np.array([model_dim(self.model)])
        arrays["_created_ns"] = np.array([time.time_ns()])
        arrays["_reason"] = np.array([str(reason)])
        arrays["_n"] = np.array([n])
        arrays["_recorded"] = np.array([
            sum(1 for label in self._pool for t in self.takes(label)
                if t.recorded)])

        path = self.path_for(gen)
        tmp = path.with_name(path.name + ".tmp")
        # os.open with an explicit 0o600 rather than open()-then-chmod: under
        # his umask the plain form creates 0664 and stays 0664 for the whole
        # of np.savez. No O_EXCL -- a tmp left by a hard crash would then wedge
        # every future save, and a store whose failure mode is "cannot write"
        # has lost the argument it exists to win.
        #
        # savez appends ".npz" to a BARE PATH, so it is handed a file handle;
        # speaker.py:288 hit that same trap and the comment there says so.
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **arrays)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                log.warning("could not remove a failed save's %s", tmp.name,
                            exc_info=True)
            raise
        self.loaded_generation = gen
        self._loaded_n = n
        self._provenance = {"format": FORMAT, "created_ns": time.time_ns(),
                            "reason": str(reason), "n": n,
                            "recorded": int(arrays["_recorded"][0]),
                            "model": self.model}
        if prune:
            self._prune()
        log.info("voice gallery saved: generation %d, %d samples over %d "
                 "label(s) (%s)", gen, n, len(self._pool),
                 ", ".join(sorted(self._pool)) or "-")
        return gen

    def _sample_count(self, generation: int) -> int:
        try:
            _pool, _takes, _consent, prov = self._read(self.path_for(generation))
        except Exception:  # noqa: BLE001 - unreadable defends nothing
            return 0
        if str(prov.get("model") or "") != self.model:
            return 0
        return int(prov.get("n") or 0)

    def _on_disk_n(self) -> int:
        """The newest parsing generation's size, read from the DISK.

        THE FIX FOR THE INCIDENT'S OWN SHAPE: the write that destroyed the
        voiceprint came from a freshly built object that had loaded nothing, so
        a purely in-memory baseline is 0 and the shrink guard abstains on
        exactly the case it exists for."""
        for gen in reversed(self.generations()):
            n = self._sample_count(gen)
            if n:
                return n
        return 0

    def _check_not_collapsed(self) -> None:
        """A pool whose samples are all the same vector is not an enrolment,
        and it is indistinguishable from a working one until the day it
        refuses somebody. See COLLAPSED_MEDIAN_COSINE for the measured gap
        this bar sits in -- and for what it does not catch."""
        for label, pool in self._pool.items():
            med = median_pairwise(pool)
            if med is None:
                continue
            if med > COLLAPSED_MEDIAN_COSINE:
                raise ValueError(
                    "refusing to save a collapsed pool for %r: its %d samples "
                    "have a median pairwise cosine of %.3f, above %.2f. His "
                    "own 14-take pool measures 0.485; a pool this tight is "
                    "one take saved several times, not several takes."
                    % (label, len(pool), med, COLLAPSED_MEDIAN_COSINE))

    def _own_generations(self, gens=None) -> List[int]:
        """The generations THIS model wrote, oldest first. An unreadable one
        counts as ours -- it may be, and the alternative is a pruning rule
        that silently protects corrupt files forever."""
        out = []
        for gen in (self.generations() if gens is None else list(gens)):
            try:
                _pool, _takes, _consent, prov = self._read(self.path_for(gen))
            except Exception:  # noqa: BLE001
                out.append(gen)
                continue
            if str(prov.get("model") or "") == self.model:
                out.append(gen)
        return out

    def _prune(self) -> None:
        """Drop the oldest generations past the window -- but NEVER the
        richest, and never another model's.

        Oldest-first alone makes the stack self-destructing: five saves of any
        size evict a good enrolment and land the store in exactly the state the
        2026-09-02 incident left. Ties go to the newest, so a steady state
        prunes as it would have anyway."""
        gens = self.generations()
        keep = max(KEEP_GENERATIONS, MIN_GENERATIONS)
        mine = self._own_generations(gens)
        for tmp in self._tmp_paths():
            try:
                _shred(tmp)
            except OSError:
                log.debug("could not remove %s", tmp.name, exc_info=True)
        gens = mine
        if len(gens) <= keep:
            return
        richest, richest_n = gens[-1], -1
        for gen in gens:
            n = self._sample_count(gen)
            if n >= richest_n:          # >= so a tie protects the NEWEST
                richest, richest_n = gen, n
        doomed = [g for g in gens if g != richest][:len(gens) - keep]
        for gen in doomed:
            try:
                _shred(self.path_for(gen))
            except OSError:
                log.debug("could not prune generation %d", gen, exc_info=True)

    def rollback(self) -> int:
        """Delete the newest generation and load the one before it; return the
        generation now loaded, or 0. The step the voiceprint did not have."""
        gens = self._own_generations()
        if len(gens) < 2:
            return 0
        # Prove there is something to fall back TO before destroying what is
        # here: an unreadable older generation defends nothing, and shredding
        # on a file COUNT is how the face gallery once emptied its directory.
        if not any(self._sample_count(g) > 0 for g in gens[:-1]):
            return 0
        _shred(self.path_for(gens[-1]))
        self.loaded_generation = 0
        self._loaded_n = 0
        return self.loaded_generation if self.load() else 0

    def purge(self) -> int:
        """Delete EVERY generation and every crashed save's tmp; return the
        count. Every file is overwritten before it is unlinked."""
        removed = 0
        for path in [self.path_for(g) for g in self.generations()] + self._tmp_paths():
            try:
                _shred(path)
                removed += 1
            except OSError:
                log.warning("could not delete voice gallery file %s",
                            path.name, exc_info=True)
        self.reset()
        self.loaded_generation = 0
        self._loaded_n = 0
        self._provenance = {}
        log.info("voice gallery purged: %d file(s) deleted", removed)
        return removed

    def purge_label(self, label: str, reason: str = "") -> dict:
        """Destroy ONE person's embeddings, everywhere on the disk.

        THE INVARIANT, and it is the whole method:

            NO FILE IS DESTROYED UNLESS THE EMBEDDINGS IT HELD, MINUS THEIRS,
            HAVE BEEN READ BACK FROM THE NEW GENERATION ON DISK.

        Not inferred from a return value, not counted in memory, not assumed
        because a save did not raise. READ BACK. facegallery.purge_label
        carries the full argument and the three separate bugs that produced
        it; this is the same method over the same file layout, and it matters
        more here rather than less -- a guest who withdraws consent has no
        other way to make their voice leave this machine.

        Anything that fails any step is counted, named and LEFT ALONE, and the
        caller stops claiming the delete was carried out.
        """
        if self.root is None:
            raise ValueError("this gallery has no root; it cannot be purged")
        label = str(label)
        out: dict = {"label": label, "generations_with": [], "removed": 0,
                     "unreadable": [], "foreign": [], "foreign_models": (),
                     "not_carried": [], "still_holding": [], "tmp_removed": 0,
                     "generation": 0, "loaded": 0, "left": 0,
                     "labels_left": (), "labels_on_disk": (), "reason": "",
                     "complete": False}

        # ------------------------------------------ 1. inventory, raw, first
        inventory: Dict[int, Dict[str, int]] = {}
        wrote_by: Dict[int, str] = {}
        unreadable: List[int] = []
        for gen in self.generations():
            counts, wrote = self._inventory(self.path_for(gen))
            if counts is None:
                unreadable.append(gen)
                continue
            inventory[gen] = counts
            wrote_by[gen] = wrote
        holds = [g for g in sorted(inventory) if inventory[g].get(label)]
        out["generations_with"] = list(holds)
        out["unreadable"] = list(unreadable)
        if not holds:
            out["tmp_removed"] = self._shred_tmps()
            return self._purge_verdict(out, label, unreadable)
        carry: Dict[str, int] = {}
        for gen in holds:
            for other, n in inventory[gen].items():
                if other != label and n:
                    carry[other] = max(carry.get(other, 0), n)

        # ---------------------------------------- 2. write the replacement
        self.load()
        self.forget(label)
        if carry and self.total():
            try:
                out["generation"] = self.save(
                    reason=reason or ("forget %s" % label),
                    allow_shrink=True, prune=False)
            except ValueError as exc:
                out["reason"] = str(exc)
                log.warning("voice gallery: not deleting %r -- what is left "
                            "could not be saved: %s", label, exc)
                return self._purge_verdict(out, label, unreadable)

        # ------------------------------------- 3. read the replacement back
        proven: Dict[str, int] = {}
        if out["generation"]:
            proven, why = self._proven_survivors(out["generation"], label)
            if why:
                out["reason"] = why
                log.warning("voice gallery: not deleting %r -- %s", label, why)
                return self._purge_verdict(out, label, unreadable)

        # ------------------------------------------------ 4. and only then
        left_alone: List[int] = []
        for gen in holds:
            if gen == out["generation"]:
                continue
            short = sorted(other for other, n in inventory[gen].items()
                           if other != label and n > proven.get(other, 0))
            if short:
                left_alone.append(gen)
                log.warning("voice gallery: generation %d also holds %s, and "
                            "the new generation does not carry them at full "
                            "count; LEFT ALONE with %r still in it",
                            gen, ", ".join(short), label)
                continue
            path = self.path_for(gen)
            if not path.exists():
                continue
            try:
                _shred(path)
                out["removed"] += 1
            except OSError:
                left_alone.append(gen)
                log.warning("could not delete voice gallery generation %d",
                            gen, exc_info=True)
        out["foreign"] = [g for g in left_alone
                          if wrote_by.get(g, self.model) != self.model]
        out["not_carried"] = [g for g in left_alone if g not in out["foreign"]]
        out["foreign_models"] = tuple(sorted({wrote_by[g]
                                              for g in out["foreign"]}))
        out["tmp_removed"] = self._shred_tmps()
        return self._purge_verdict(out, label, unreadable)

    def _proven_survivors(self, generation: int, label: str):
        """``({label: samples}, "")`` READ BACK FROM DISK, or ``({}, why)``.

        The step that makes the invariant an invariant. Everything upstream --
        the return of ``save``, the pool in memory, the number in the
        provenance -- is a report ABOUT a write, and the failure this store
        exists for is a write that succeeded and left the wrong bytes."""
        path = self.path_for(generation)
        counts, wrote = self._inventory(path)
        if counts is None:
            return {}, ("the new generation %d cannot be read back from disk"
                        % generation)
        if wrote != self.model:
            return {}, ("the new generation %d reads back as %r, not %r"
                        % (generation, wrote, self.model))
        if counts.get(label):
            return {}, ("the new generation %d still holds %r"
                        % (generation, label))
        try:
            pool, _takes, _consent, prov = self._read(path)
        except Exception as exc:  # noqa: BLE001 - any failure is a refusal
            return {}, ("the new generation %d does not load: %s"
                        % (generation, exc))
        if str(prov.get("model") or "") != self.model:
            return {}, ("the new generation %d loads as %r, not %r"
                        % (generation, prov.get("model"), self.model))
        if pool.get(label):
            return {}, ("the new generation %d loads with %r still in it"
                        % (generation, label))
        return {k: len(v) for k, v in pool.items() if v}, ""

    def _purge_verdict(self, out: dict, label: str,
                       unreadable: List[int]) -> dict:
        """Read the disk AFTER the shredding and say whether it is finished.

        The other direction of the promise: it must be impossible to report
        success while their vectors are still on the disk."""
        still: List[int] = []
        unknown = list(unreadable)
        for gen in self.generations():
            counts, _wrote = self._inventory(self.path_for(gen))
            if counts is None:
                if gen not in unknown:
                    unknown.append(gen)
                continue
            if counts.get(label):
                still.append(gen)
        out["unreadable"] = sorted(set(unknown))
        out["still_holding"] = still
        self.load()
        out["loaded"] = self.loaded_generation
        out["left"] = self.total()
        out["labels_left"] = self.labels()
        out["labels_on_disk"] = self.disk_labels()
        out["complete"] = not (still or out["unreadable"] or self.leftovers()
                               or out["reason"])
        log.info("voice gallery: %r removed from %d generation(s); %d "
                 "embeddings over %d label(s) left; %d unreadable, %d foreign, "
                 "%d not carried forward; %d still hold %r; complete: %s",
                 label, out["removed"], out["left"], len(out["labels_left"]),
                 len(out["unreadable"]), len(out["foreign"]),
                 len(out["not_carried"]), len(still), label, out["complete"])
        return out

    def _shred_tmps(self) -> int:
        gone = 0
        for tmp in self._tmp_paths():
            try:
                _shred(tmp)
                gone += 1
            except OSError:
                log.warning("could not delete voice gallery leftover %s",
                            tmp.name, exc_info=True)
        return gone

    def drop_generations(self, generations) -> int:
        """Delete exactly these generations; return how many went. The last
        step of REPLACING, never of destroying: the caller already holds a
        number for a new generation that is safely on disk."""
        wanted = {int(g) for g in generations}
        removed = 0
        for gen in self.generations():
            if gen not in wanted:
                continue
            try:
                _shred(self.path_for(gen))
                removed += 1
            except OSError:
                log.warning("could not delete voice gallery generation %d",
                            gen, exc_info=True)
        if removed:
            log.info("voice gallery: %d superseded generation(s) deleted",
                     removed)
        return removed

    # ------------------------------------------------------- the migration
    def migrate_voiceprint(self, label: str, path: Optional[Path] = None,
                           reason: str = "") -> dict:
        """Carry an existing single-speaker ``voiceprint.npz`` in under one
        label. HE DOES NOT PAY FOR THIS FEATURE WITH A RE-ENROLMENT.

        Same encoder, same 192 dimensions, same silence-trimmed pipeline, so
        the vectors are valid exactly as they stand: they move, they are not
        recomputed. He re-enrolled his face this week and should not be asked
        to sit through eight more takes to get a store he did not ask for.

        ONLY FORMAT 2 MIGRATES. Format 1 predates ``trim_silence``: those
        embeddings were pooled with the silence of fixed-length enrolment
        takes while probes are trimmed now, and the asymmetry measurably
        lowers genuine scores. Carrying them across would import a known-bad
        pool under a new name and hide it behind a fresh format number, so
        format 1 is REFUSED and speaker.py's existing re-enrol message stands.

        REVERSIBLE BY CONSTRUCTION, and the reversal is the point rather than
        a courtesy. Nothing is read from ``voiceprint.npz`` but numbers, and
        nothing at all is written to it -- it stays byte for byte what it was,
        which is what makes it the rollback. Undoing this is deleting the
        generation it wrote.

        Returns numbers so a script can print them and a test can read them.
        """
        src = Path(path) if path is not None else PATHS.VOICEPRINT
        out = {"ok": False, "label": str(label), "source": str(src),
               "format": 0, "found": 0, "migrated": 0, "dropped": 0,
               "generation": 0, "why": ""}
        if not _LABEL_RE.match(str(label or "")):
            out["why"] = ("%r is not a label the registry, the face gallery "
                          "and this store can all hold" % (label,))
            return out
        if self.root is None:
            out["why"] = "this gallery has no root; it cannot be saved"
            return out
        if not src.exists():
            out["why"] = "there is no voiceprint at %s to migrate" % src
            return out
        # NEVER OVER AN EXISTING ENROLMENT. A second run must not append 14
        # more copies of the same takes, which would double the label's weight
        # in its own centroid and quietly make every later margin wrong.
        if label in self.disk_labels():
            out["why"] = ("%s already has embeddings in the voice gallery; "
                          "migrating again would store the same takes twice"
                          % label)
            return out
        try:
            data = np.load(src)
            names = list(data.files)
            fmt = int(data["_format"][0]) if "_format" in names else 1
        except Exception as exc:  # noqa: BLE001 - any failure is a refusal
            out["why"] = "%s could not be read (%s)" % (src, type(exc).__name__)
            return out
        out["format"] = fmt
        if fmt != 2:
            out["why"] = (
                "%s is format %d and only format 2 migrates. Format 1 predates "
                "silence trimming: those embeddings were pooled with the "
                "silence of fixed-length takes and score low against trimmed "
                "probes, so they would arrive here already broken. Re-enrol "
                "with scripts/enroll_voice.py --reset instead." % (src, fmt))
            log.warning("voice gallery: %s", out["why"])
            return out
        keys = sorted(k for k in names if k.startswith("emb_"))
        out["found"] = len(keys)
        if not keys:
            out["why"] = "%s holds no embeddings" % src
            return out
        staged = []
        for key in keys:
            arr = np.asarray(data[key], dtype=np.float32).ravel()
            why = degenerate_reason(arr, self.model)
            if why:
                out["dropped"] += 1
                log.warning("voice gallery migration: dropping %s (%s)",
                            key, why)
                continue
            staged.append(arr)
        if not staged:
            out["why"] = "%s holds no usable embeddings" % src
            return out
        before = dict(self._pool), dict(self._takes), dict(self._consent)
        for arr in staged:
            self.add(label, arr, src="legacy",
                     note="migrated from voiceprint.npz format 2")
        # The owner's own pool is his own consent; identity.Person carries the
        # same distinction ("owner" vs "typed") and face_enrol.consent draws it
        # in exactly this place.
        if not self.consent(label):
            self.set_consent(label, "owner")
        try:
            out["generation"] = self.save(
                reason=reason or ("migrated %d takes from voiceprint.npz "
                                  "format 2" % len(staged)))
        except ValueError as exc:
            self._pool, self._takes, self._consent = before
            out["why"] = str(exc)
            log.warning("voice gallery: migration refused -- %s", exc)
            return out
        out["migrated"] = len(staged)
        out["ok"] = True
        log.info("voice gallery: migrated %d take(s) from %s as %r into "
                 "generation %d; %s is untouched and stays the rollback",
                 len(staged), src.name, label, out["generation"], src.name)
        return out

    def provenance(self) -> dict:
        return dict(self._provenance)


def default_gallery(model: Optional[str] = None) -> VoiceGallery:
    """The store, wherever PATHS says it is.

    ``PATHS.VOICE_GALLERY`` honours ``JARVIS_VOICE_GALLERY``, which
    tests/conftest.py forces into a throwaway directory -- so importing this
    in a test cannot reach anybody's real enrolment."""
    return VoiceGallery(root=PATHS.VOICE_GALLERY,
                        model=model or DEFAULT_MODEL)
