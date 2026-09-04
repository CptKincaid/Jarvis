"""The face gallery: enrolled face embeddings, and the only biometric
artefact this feature puts on disk.

A voiceprint is a recording of how he sounds; a face embedding is a
measurement of his face. Both are data about one specific person and neither
can be re-issued if it leaks, so this store is written to a stricter standard
than the rest of the app -- and the standard comes from an incident, not from
principle.

WHAT WENT WRONG ON 2026-09-02, because this module is the answer to it.
A test built a real ``SpeakerVerifier`` and called ``enroll_from_audio``.
``save()`` (jarvis/speaker.py:276) writes the module-global voiceprint path,
so his six-sample pool was replaced with two copies of the fixture's constant
vector -- every element 0.07216878. The copy kept beside it held only those
fixtures. The original was UNRECOVERABLE and he re-enrolled by hand. See
jarvis/config.py:46-55 and the JARVIS_VOICEPRINT paragraph in
tests/conftest.py.

``speaker.save`` is *atomic* -- tmp file plus ``os.replace``. Atomicity
protects against a crash in the middle of a write and against nothing else.
The write that destroyed the voiceprint succeeded. So four things here that
atomicity does not give you:

1. **Generational.** A save never overwrites. It writes ``gen-00002.npz``
   beside ``gen-00001.npz``, and ``load()`` takes the newest that parses. A
   bad write costs one generation, not the enrolment; ``rollback()`` undoes
   it. Old generations ARE the backup -- there is no separate ".bak" that can
   quietly come to hold the same bad data the live file does.

   That claim was FALSE in the first version of this file and the correction
   is ``_prune()``. Keeping the newest five oldest-first means five bad writes
   evict the good enrolment and leave exactly the state the incident left --
   measured 2026-09-02: one 6-sample enrolment plus six 2-sample saves left
   generations [4,5,6,7,8], every one holding n=2 and the enrolment gone. So
   the RICHEST generation is never pruned. A backup you can delete by
   repetition is not a backup.
2. **A degenerate-vector guard.** The fixture was a constant vector. A real
   embedding never is (measured on this box 2026-09-02: cv2.FaceRecognizerSF
   returns 128 float32 with L2 = 10.41 and per-element spread), so the exact
   shape of that accident is refused at ``add()``.
3. **A collapsed-pool guard.** Two copies of one vector is not an enrolment,
   and is what the incident LEFT BEHIND. ``save()`` refuses it.
4. **A shrink guard.** Six samples became two and nothing objected. Dropping
   samples now needs ``allow_shrink=True``, which the enrolment script says
   out loud and a stray test does not.

   The baseline is the LARGER of what this object loaded and what is already
   on disk. Comparing only against what this object loaded is the version
   that misses the incident: ``enroll_from_audio`` built a fresh verifier and
   saved without ever loading, so an in-memory baseline is 0 and the guard
   abstains on precisely its own motivating case (measured 2026-09-02: a
   never-loaded gallery wrote a 2-sample generation over a 6-sample one
   without a word).

And, belt and braces, the store obeys the test firewall the same way the
voiceprint now does: ``PATHS.FACE_GALLERY`` reads ``JARVIS_FACE_GALLERY``,
which tests/conftest.py forces into a throwaway directory before any jarvis
import. A firewall and a recoverable format are different defences and this
data warrants both -- the voiceprint had neither on the day it was lost.

FORMAT. One ``.npz`` per generation: ``emb_<label>_<nnnn>`` float32 arrays,
plus ``_format``, ``_created_ns`` and ``_reason``. Labels live in the key
because a face gallery holds more than one person the moment a second person
sits down -- distinguishing "someone" from "him" is the point.

WHAT A TAKE SAYS IT WAS, and why it is a KEY and not a format bump. He asked
for "a note on what i am doing in the take" -- looking at my phone, looking
away, with glasses -- because when a match scores badly the only useful
question is WHICH POSE is weak, and 128 floats cannot answer it. So each
sample may carry a ``note_<label>_<nnnn>`` string and a ``yaw_<label>_<nnnn>``
float beside its embedding.

They are OPTIONAL KEYS AT THE SAME ``_format``, and that is the whole design
constraint. ``_read`` REFUSES a format number it does not know (four lines
below the paragraph above, and correctly -- reading a newer pool as if it
were this one is how embeddings silently stop comparing). So bumping FORMAT
for notes would have made his live enrolment -- 13 embeddings written at
23:29 on 2026-09-02, verified at 120 of 120 frames matched -- unreadable by
the build that added the feature, which is precisely the class of loss this
module exists to prevent. A missing note key is not an error; it is a take
that predates the notes, and ``Take.recorded`` is False so nothing downstream
can mistake "no record" for "frontal". A note key is paired to its embedding
by INDEX, so a vector dropped for being degenerate takes its note with it --
otherwise every note after the dropped one describes the wrong face.

DELETING ONE PERSON is ``purge_label()``, and it is not ``forget()`` plus a
save. A save writes a new generation; the OLD generations still hold her, one
``rollback()`` away and still on the disk. Consent withdrawn has to mean the
embeddings go, so ``purge_label`` writes what is left as a new generation and
then SHREDS every generation that held her -- including the ones that are
unreadable, because nothing can prove those do not hold her either.

DELETION is ``purge()``: every generation, not just the newest, AND any
``.tmp`` a crashed save left behind -- which is a full set of embeddings under
a name the generation pattern does not match, so the first version of purge()
reported success and left one on disk. A store whose "delete" leaves an older
copy of his face on disk has not deleted anything. Every deletion in this
module goes through ``_shred``: the bytes are overwritten and then unlinked,
because unlink alone drops the directory entry and leaves the vectors in the
extents. Read ``_shred`` for the limit of that -- it is a filesystem-level
erase, not a device-level one, and the command says so in those words.

Nothing here imports cv2, torch or a model. It is arithmetic over arrays, so
it runs in the test suite with no camera, no display and no GPU.
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

log = get_logger("facegallery")

# NOT BUMPED FOR THE MODEL SWAP, AND THAT IS THE WHOLE POINT. ``_read``
# refuses a format it does not recognise, so bumping this would make his
# existing SFace generations UNREADABLE -- which would take his ability to
# revert with them. Instead the model is carried in an OPTIONAL ``_model``
# key: a generation without one was written by SFace, because nothing else
# has ever written this store.
FORMAT = 1

# ------------------------------------------------------------ which model
# WHICH MODEL PRODUCED A VECTOR IS PART OF THE VECTOR. An ArcFace 512-vector
# and an SFace 128-vector are not weakly comparable, they are not comparable:
# the cosine between them is not a worse measurement of the same thing, it is
# a measurement of nothing. Before this key existed the store recorded only
# the numbers, so a mixed gallery would have scored his own face against
# noise and reported it as a low match -- "recognition got worse" with no way
# to find out why. Every read, every write and every comparison here now names
# the model, and a cross-model comparison is refused BY NAME rather than
# scored.
SFACE_MODEL = "sface"
ARCFACE_MODEL = "arcface_mbf"
# SFace (face_recognition_sface_2021dec.onnx, Apache-2.0) returns 128 float32.
# Measured on this box 2026-09-02: shape (1, 128), L2 norm 10.41 -- NOT unit
# length, which is why every comparison here normalises rather than dotting.
# ArcFace w600k_mbf returns 512 and jarvis/faceinsight.normalise unit-lengths
# it at the source.
MODEL_DIMS = {SFACE_MODEL: 128, ARCFACE_MODEL: 512}
# What a generation with no ``_model`` key was written by. This is not a
# guess: the key was added on 2026-09-03 and SFace is the only recogniser
# that had ever written this store.
LEGACY_MODEL = SFACE_MODEL
DEFAULT_MODEL = SFACE_MODEL
EMBED_DIM = MODEL_DIMS[SFACE_MODEL]


def model_dim(model: str) -> int:
    """How many floats a vector from ``model`` has, or ValueError by name."""
    try:
        return MODEL_DIMS[str(model)]
    except KeyError:
        raise ValueError(
            "unknown embedding model %r; known: %s"
            % (model, ", ".join(sorted(MODEL_DIMS)))) from None
# Five is enough history to undo a mistake noticed a few enrolments later and
# small enough that the whole store stays under a megabyte at 128 floats a
# sample. Never pruned below two: one generation is no backup at all.
KEEP_GENERATIONS = 5
MIN_GENERATIONS = 2
# OpenCV's documented SFace cosine threshold for "same person" is 0.363; this
# store does not enforce it (the consumer decides how sure it needs to be) but
# it is the number a caller should start from.
SFACE_COSINE_SAME = 0.363
# AND THERE IS NO SUCH NUMBER FOR ARCFACE HERE, on purpose. 0.363 is OpenCV's
# published figure for SFace's vectors; carrying it across to a different
# model's would be a threshold that has stopped meaning anything, which is the
# exact failure this module's cross-model refusal exists to prevent. It cannot
# be measured without his face, so ``scripts/face_model_compare.py`` is the
# instrument and he is the one who runs it. None means UNMEASURED, and every
# consumer must treat it as "say so", not as "use zero".
ARCFACE_COSINE_SAME = None
MODEL_COSINE_SAME = {SFACE_MODEL: SFACE_COSINE_SAME,
                     ARCFACE_MODEL: ARCFACE_COSINE_SAME}


def cosine_same(model: str):
    """The published "same person" cosine for a model, or None if unmeasured."""
    return MODEL_COSINE_SAME.get(str(model))

_GEN_RE = re.compile(r"^gen-(\d{5})\.npz$")
# A crashed save leaves gen-00002.npz.tmp, which _GEN_RE does not match -- so
# the first purge() reported success and left a full set of his embeddings on
# disk under a name nothing looked for. Deletion has to mean deletion.
_TMP_RE = re.compile(r"^gen-(\d{5})\.npz\.tmp$")
_KEY_RE = re.compile(r"^emb_(.+)_(\d{4})$")
# A label goes into an npz key and a log line, so keep it boring.
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")
# A note goes into an npz key, a printed report he PASTES to somebody, and a
# log line. One printable line, and short enough that a whole gallery's worth
# of them still reads as a table.
NOTE_MAX = 120


@dataclass(frozen=True)
class Take:
    """What one sample was, beyond the 128 floats.

    ``note`` is his own words for the pose -- "looking at my phone" -- and
    ``yaw_deg`` is what the head model measured while it was captured.

    ``yaw_deg`` IS None RATHER THAN 0.0 WHEN NOTHING WAS RECORDED, and the
    difference matters more than it looks: his live generation carries no
    angles at all, and a 0.0 default would report 13 perfectly frontal takes
    that were never measured -- turning "this gallery cannot say what it
    covers" into a confident and wrong "it is fully covered frontally". The
    same three-valued contract ``jarvis/roomsensor.py`` writes down for the
    radar: absent is not zero."""

    note: str = ""
    yaw_deg: Optional[float] = None

    @property
    def recorded(self) -> bool:
        """Did anything about this take get written down at all?"""
        return bool(self.note) or self.yaw_deg is not None

    def as_dict(self) -> dict:
        return {"note": self.note, "yaw_deg": self.yaw_deg}


def label_ok(label) -> bool:
    """Is this a name ``add()`` will accept?

    Public because the enrolment script has to refuse a bad ``--label`` at
    the ARGUMENT, before a minute in front of the camera, and reaching into
    ``_LABEL_RE`` from outside would make the rule two rules that can drift."""
    return bool(_LABEL_RE.match(str(label or "")))


def clean_note(text) -> str:
    """One printable line, at most ``NOTE_MAX`` characters.

    A note is free text: he types it at a prompt, or it arrives from a voice
    command through the entry point. It then lands in an npz key's value, in
    a report he pastes into a chat window, and in a log line. Newlines would
    break the report's table, control characters would break the terminal it
    is pasted into, and anything that is not a string at all is refused by
    being turned into one -- ``str(np.zeros((4, 4)))`` is a string and is
    then capped like any other, so a note can never smuggle an array onto the
    disk under a key the numbers-only checker does not inspect."""
    s = "" if text is None else str(text)
    s = "".join(c if (c.isprintable() and c != "\x00") else " " for c in s)
    return " ".join(s.split())[:NOTE_MAX]


def clean_yaw(value) -> Optional[float]:
    """A finite angle, or None meaning "not recorded".

    A NaN yaw is what a failed head model produces, and it would compare
    False against every bucket edge -- landing the take in no bucket while
    still counting as recorded. None says the true thing instead."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _shred(path: Path) -> None:
    """Overwrite a generation's bytes, then unlink it.

    ``unlink`` removes the directory ENTRY. The extents that held the
    128-float vectors stay on the device until the filesystem reuses them, so
    a "delete" that only unlinks leaves a measurement of his face on the disk
    while the command says it is gone. The files are ~6 KB, so overwriting
    first costs nothing measurable.

    SAY THE LIMIT OUT LOUD RATHER THAN LEAVE IT IMPLIED. This makes the bytes
    unreachable THROUGH THE FILESYSTEM. It is not a device-level erase: on a
    copy-on-write filesystem, on a journalled one that already wrote the
    block elsewhere, and on any SSD (this box) whose controller remaps rather
    than rewrites, the old blocks can survive an in-place overwrite. What the
    caller is entitled to claim is exactly what this does -- which is why
    scripts/face_enrol.py's --delete says "overwritten and unlinked" and not
    "off the disk".

    Raises ``OSError`` from the unlink so existing callers keep their own
    handling; a failure to OVERWRITE is logged and the unlink still happens,
    because leaving the file in place would be strictly worse.
    """
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


def cosine(a, b) -> float:
    """Cosine similarity, and 0.0 rather than NaN for a zero vector.

    A zero vector is what a failed crop or an all-black frame produces. NaN
    would propagate through every comparison and, because ``NaN > threshold``
    is False everywhere, would read as "no match" in one place and poison a
    mean in another. Zero is the honest answer: no evidence."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or not np.isfinite(na) or not np.isfinite(nb):
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def degenerate_reason(vec, model: str = DEFAULT_MODEL) -> str:
    """"" if this could be a face embedding from ``model``, else why not.

    The three shapes that have actually appeared: the wrong dimension (a
    different model's output), a non-finite element (a crop of nothing), and a
    CONSTANT vector -- which is what the fixture that destroyed his voiceprint
    contained, and which no real embedding is.

    THE DIMENSION CHECK IS NOW THE MODEL CHECK'S LAST LINE, not its first.
    512 floats where 128 belong is caught here and named; 512 floats where 512
    belong but from the wrong 512-D model is caught upstream by the ``_model``
    key, because arithmetic cannot see that one."""
    dim = model_dim(model)
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size != dim:
        return ("wrong dimension: %d, expected %d for %s"
                % (arr.size, dim, model))
    if not np.all(np.isfinite(arr)):
        return "not finite"
    if float(np.linalg.norm(arr)) == 0.0:
        return "all zero"
    # A real SFace vector's elements spread widely; a fixture's do not. The
    # bar is deliberately far below any plausible embedding so it can only
    # ever catch the accident, never a real face.
    if float(np.std(arr)) < 1e-6:
        return "constant vector (std %.2e) -- this is a fixture, not a face" % float(np.std(arr))
    return ""


def _take_for(data, have, label: str, idx: str) -> Take:
    """The note and yaw for one stored embedding, and never an exception.

    Split out so the failure is contained to ONE take: see the block in
    ``_read``. A key that will not read costs its own note, is logged, and
    leaves a blank ``Take`` -- which is exactly the state every take his
    2026-09-02 generation is in anyway, so nothing downstream is surprised
    by it."""
    note, yaw = "", None
    note_key = "note_%s_%s" % (label, idx)
    yaw_key = "yaw_%s_%s" % (label, idx)
    if note_key in have:
        try:
            note = clean_note(str(data[note_key][0]))
        except Exception:  # noqa: BLE001 - a bad note costs the note
            log.warning("face gallery: unreadable %s; the take keeps its "
                        "embedding and loses its note", note_key,
                        exc_info=True)
    if yaw_key in have:
        try:
            yaw = clean_yaw(data[yaw_key][0])
        except Exception:  # noqa: BLE001
            log.warning("face gallery: unreadable %s; the take keeps its "
                        "embedding and loses its angle", yaw_key,
                        exc_info=True)
    return Take(note, yaw)


class FaceGallery:
    """Enrolled faces, held in memory, persisted as numbered generations.

    ``root=None`` is a live, unsaveable gallery: useful for a caller that only
    wants to match against something it just built, and used by the tests that
    check the guards without touching a filesystem at all."""

    def __init__(self, root: Optional[Path] = None,
                 model: str = DEFAULT_MODEL):
        self.root: Optional[Path] = None if root is None else Path(root)
        # WHAT THIS GALLERY IS FOR. Every vector added, loaded, saved and
        # matched has to have come from this model; anything else is refused
        # by name. Defaulting to SFace keeps every existing caller and every
        # existing test writing exactly the file it wrote before -- the
        # production path names the model explicitly through
        # ``default_gallery``.
        self.model = str(model)
        model_dim(self.model)          # raise now, not on his enrolment
        # Generations on disk written by SOMETHING ELSE. Not an error and not
        # deleted: they are his old enrolment, and they are what he reverts
        # to. ``load()`` fills this in so a caller can say the true sentence
        # -- "13 sface takes are on disk, none for arcface_mbf, re-enrol".
        self.foreign_generations: Dict[int, str] = {}
        self._pool: Dict[str, List[np.ndarray]] = {}
        # Kept in lockstep with _pool, index for index. ``takes()`` pads
        # rather than trusting that, because a mismatch would attach one
        # sample's note to another sample's face.
        self._takes: Dict[str, List[Take]] = {}
        self.loaded_generation = 0
        self._loaded_n = 0            # the size the shrink guard compares to
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

    def takes(self, label: str) -> List[Take]:
        """What each of this label's samples says it was, index for index
        with ``embeddings(label)``.

        Padded to the pool's length with blank takes rather than returning a
        shorter list: every caller zips the two, and a short list would
        silently drop the last samples from a coverage count -- reporting
        better coverage than the gallery has, which is the one direction this
        feature must not fail in."""
        n = len(self._pool.get(label, ()))
        got = list(self._takes.get(label, ()))
        if len(got) < n:
            got = got + [Take()] * (n - len(got))
        return got[:n]

    def reset(self) -> None:
        """Empty the in-memory pool. Touches no disk: the generations stay,
        which is what makes "enrol again from scratch" a safe thing to do."""
        self._pool = {}
        self._takes = {}

    def add(self, label: str, vec, note="", yaw_deg=None) -> None:
        """Add one embedding and what it was, or raise ValueError saying why
        not.

        ``note`` and ``yaw_deg`` both default to "nothing recorded", so every
        existing caller keeps writing exactly the file it wrote before. The
        take is appended in the SAME statement region as the vector and only
        after every refusal above has passed, so the two lists cannot come
        apart on a rejected sample."""
        if not _LABEL_RE.match(label or ""):
            raise ValueError("bad label %r: lowercase letters, digits, - and _" % label)
        why = degenerate_reason(vec, self.model)
        if why:
            raise ValueError("refusing a degenerate embedding: %s" % why)
        arr = np.asarray(vec, dtype=np.float32).ravel().copy()
        self._pool.setdefault(label, []).append(arr)
        self._takes.setdefault(label, []).append(
            Take(clean_note(note), clean_yaw(yaw_deg)))

    def forget(self, label: str) -> int:
        """Drop one person from the in-memory pool; save() commits it.

        IN MEMORY ONLY, AND THAT IS NOT A DELETE. The generations on disk
        still hold them, and one ``rollback()`` brings them back. Removing a
        person because they withdrew consent is ``purge_label()``."""
        gone = len(self._pool.pop(label, ()))
        self._takes.pop(label, None)
        return gone

    def match(self, vec) -> Tuple[str, float]:
        """The nearest enrolled label and its cosine, or ("", 0.0).

        Nearest by the pool's BEST sample rather than by a centroid: a
        centroid of him in glasses and him without is a face that does not
        exist, and the same averaging is why the voiceprint's own pool keeps
        its members (jarvis/speaker.py:249-262)."""
        if degenerate_reason(vec, self.model):
            return "", 0.0
        best_label, best = "", 0.0
        for label, pool in self._pool.items():
            for emb in pool:
                s = cosine(vec, emb)
                if s > best:
                    best_label, best = label, s
        return best_label, best

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
        """Any ``gen-NNNNN.npz.tmp`` a crashed save left. These hold a full
        set of embeddings and are invisible to ``_GEN_RE``."""
        if self.root is None or not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if _TMP_RE.match(p.name))

    def leftovers(self) -> List[str]:
        """The names of any crashed-save ``gen-NNNNN.npz.tmp`` files.

        Public because a caller that has just deleted somebody has to be able
        to CHECK, and ``generations()`` -- which every read-back walks -- is
        blind to these by design (``_GEN_RE`` does not match them). A tmp
        holds a whole pool, so one surviving is a failed deletion, not an
        untidy directory."""
        return sorted(p.name for p in self._tmp_paths())

    def load(self, generation: Optional[int] = None) -> bool:
        """Load one generation, defaulting to the newest that parses.

        Falling back down the stack is the whole reason the stack exists: a
        truncated or corrupt newest file must cost the last enrolment, not the
        enrolment."""
        wanted = [generation] if generation else list(reversed(self.generations()))
        self.foreign_generations = {}
        for gen in wanted:
            try:
                pool, takes, prov = self._read(self.path_for(gen))
            except Exception:
                log.warning("face gallery generation %d unreadable; "
                            "falling back to the one before", gen, exc_info=True)
                continue
            wrote = str(prov.get("model") or LEGACY_MODEL)
            if wrote != self.model:
                # A CROSS-MODEL LOAD IS REFUSED, NOT SCALED, NOT TRUNCATED AND
                # NOT SILENTLY SKIPPED. Its vectors measure a different thing;
                # cosine against them is meaningless rather than merely weak.
                # The file is left exactly where it is -- it is his previous
                # enrolment and the thing he reverts to.
                self.foreign_generations[gen] = wrote
                log.warning(
                    "face gallery generation %d was written by %r and this "
                    "gallery is %r. NOT comparing across models: the cosine "
                    "between them measures nothing. Those %d sample(s) stay "
                    "on disk untouched.",
                    gen, wrote, self.model, int(prov.get("n") or 0))
                continue
            self._pool = pool
            self._takes = takes
            self.loaded_generation = gen
            self._loaded_n = sum(len(v) for v in pool.values())
            self._provenance = prov
            log.info("face gallery loaded: generation %d, %d samples over %d "
                     "labels, model %s", gen, self._loaded_n, len(pool),
                     self.model)
            return True
        if self.foreign_generations:
            log.warning("face identity is OFF: %s", self.reenrol_message())
        return False

    # -------------------------------------------------- the migration line
    def foreign_sample_count(self) -> int:
        """How many samples sit on disk under a DIFFERENT model.

        Read from the files rather than remembered, because the caller that
        needs this number is the one that has just failed to load anything.
        """
        total = 0
        for gen in self.foreign_generations:
            try:
                _pool, _takes, prov = self._read(self.path_for(gen))
            except Exception:  # noqa: BLE001 - unreadable defends nothing
                continue
            total += int(prov.get("n") or 0)
        return total

    def reenrol_message(self) -> str:
        """ONE LINE saying exactly what he has to do, for the startup log.

        Not "identity unavailable". Not a stack trace. The failure this
        sentence prevents is the one where recognition quietly stops working
        after a model change and the log says something true but useless.
        """
        others = sorted({m for m in self.foreign_generations.values()})
        n = self.foreign_sample_count()
        return ("nothing is enrolled for %s (%d sample(s) on disk from %s, "
                "kept, not deleted). Say \"enrol my face\" or run "
                "scripts/face_enrol.py to re-enrol; or set "
                "camera.face_backend to \"opencv\" to go back to the old "
                "models and your existing enrolment."
                % (self.model, n, ", ".join(others) or "an older model"))

    def _read(self, path: Path):
        data = np.load(path)
        fmt = int(data["_format"][0]) if "_format" in data.files else 0
        if fmt != FORMAT:
            # Reading a newer pool as if it were this one is how embeddings
            # silently stop comparing; speaker.py:264-271 warns about exactly
            # this for the voiceprint. Refuse, do not guess.
            raise ValueError("face gallery format %d, this build reads %d" % (fmt, FORMAT))
        # THE MODEL IS READ BEFORE THE VECTORS, because it decides what a
        # valid vector looks like. A generation with no ``_model`` predates
        # the key and was written by SFace; see LEGACY_MODEL.
        wrote = (str(data["_model"][0]) if "_model" in data.files
                 else LEGACY_MODEL)
        try:
            model_dim(wrote)
        except ValueError:
            # A model this build has never heard of. Its vectors cannot be
            # validated, so none are loaded -- but the generation is REPORTED
            # rather than treated as corrupt, because "written by a newer
            # build" and "truncated" want opposite responses and destroying
            # the wrong one is unrecoverable.
            log.warning("face gallery %s was written by unknown model %r; "
                        "loading no vectors from it and leaving it alone",
                        path.name, wrote)
            return {}, {}, {"format": fmt, "created_ns": 0, "reason": "",
                            "n": 0, "recorded": 0, "model": wrote}
        pool: Dict[str, List[np.ndarray]] = {}
        takes: Dict[str, List[Take]] = {}
        have = set(data.files)
        for key in sorted(data.files):
            m = _KEY_RE.match(key)
            if not m:
                continue
            arr = np.asarray(data[key], dtype=np.float32).ravel()
            if degenerate_reason(arr, wrote):
                # A stored vector that cannot be a face is not loaded: it
                # would drag every future match toward itself. ITS NOTE GOES
                # WITH IT -- the two lists are paired by position, so keeping
                # the note of a dropped vector shifts every note after it
                # onto the wrong face.
                log.warning("face gallery: dropping %s (%s)", key,
                            degenerate_reason(arr, wrote))
                continue
            label, idx = m.group(1), m.group(2)
            pool.setdefault(label, []).append(arr)
            # A COSMETIC FIELD MAY NOT COST THE EMBEDDINGS. The note and the
            # yaw are what a take was DOING; the vector is the enrolment. A
            # zero-length or otherwise malformed note_/yaw_ key used to raise
            # out of here, which _read turns into "this generation is
            # unreadable" -- one bad string costing thirteen faces, and (with
            # purge_label) an unreadable generation is the kind that gets
            # destroyed. The pre-existing contract two lines up ("if not m:
            # continue") was already to TOLERATE a key this build does not
            # understand; this restores it for the keys this build added.
            takes.setdefault(label, []).append(
                _take_for(data, have, label, idx))
        prov = {"format": fmt,
                "created_ns": int(data["_created_ns"][0]) if "_created_ns" in data.files else 0,
                "reason": str(data["_reason"][0]) if "_reason" in data.files else "",
                "n": sum(len(v) for v in pool.values()),
                "recorded": sum(1 for ts in takes.values()
                                for t in ts if t.recorded),
                "model": wrote}
        return pool, takes, prov

    def save(self, reason: str, allow_shrink: bool = False) -> int:
        """Write the pool as the NEXT generation; return its number.

        ``reason`` is provenance, not decoration: when a gallery turns out to
        be wrong, the only question that matters is what wrote it, and on
        2026-09-02 nothing on disk could answer that."""
        if self.root is None:
            raise ValueError("this gallery has no root; it cannot be saved")
        n = self.total()
        if n == 0:
            raise ValueError("refusing to save an empty gallery")
        self._check_not_collapsed()
        baseline = max(self._loaded_n, self._on_disk_n())
        if not allow_shrink and baseline and n < baseline:
            raise ValueError(
                "refusing to shrink the gallery from %d samples to %d; pass "
                "allow_shrink=True if you mean it (this guard exists because "
                "the voiceprint went 6 -> 2 unnoticed on 2026-09-02)"
                % (baseline, n))

        # mode=0o700 on the mkdir itself, not only the chmod after it: with
        # his umask of 0002 a plain mkdir creates 0775 and stays group- and
        # world-readable until the chmod lands. A window is small, not absent,
        # and this is the one module whose stated job is the stricter standard.
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)     # belt and braces for a pre-existing dir
        gen = (self.generations() or [0])[-1] + 1
        arrays: Dict[str, np.ndarray] = {}
        for label, pool in self._pool.items():
            tks = self.takes(label)
            for i, emb in enumerate(pool):
                arrays["emb_%s_%04d" % (label, i)] = emb
                # WRITTEN ONLY WHEN THERE IS SOMETHING TO WRITE, so a pool
                # with no notes produces byte-for-byte the file it produced
                # before this feature existed -- which is what makes "his
                # generation still loads" true in both directions.
                if tks[i].note:
                    arrays["note_%s_%04d" % (label, i)] = \
                        np.array([tks[i].note])
                if tks[i].yaw_deg is not None:
                    arrays["yaw_%s_%04d" % (label, i)] = \
                        np.array([float(tks[i].yaw_deg)])
        arrays["_format"] = np.array([FORMAT])
        arrays["_created_ns"] = np.array([time.time_ns()])
        arrays["_reason"] = np.array([str(reason)])
        # The key that makes a mixed store safe. Written on every save from
        # now on; its ABSENCE is what identifies the pre-2026-09-03 SFace
        # generations, so it must never be written as an empty string.
        arrays["_model"] = np.array([str(self.model)])

        path = self.path_for(gen)
        tmp = path.with_name(path.name + ".tmp")
        # savez appends ".npz" to a bare path, so hand it a file handle and
        # keep the exact tmp name -- the same trap speaker.py:288-292 hit.
        #
        # os.open with an explicit 0o600 rather than open() then chmod: under
        # his umask of 0002 the plain form creates the file 0664 and it stays
        # 0664 for the whole of np.savez, which for 12 embeddings is not an
        # instant. chmod-after closes the door on a file others could already
        # have opened. No O_EXCL: a tmp left by an earlier hard crash would
        # then wedge every future save, and a store whose failure mode is
        # "cannot write" has lost the argument it exists to win.
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **arrays)
            os.chmod(tmp, 0o600)       # if it already existed, at any mode
            os.replace(tmp, path)
        except Exception:
            # A half-written tmp holds real embeddings. Leaving it is both a
            # leak and a file purge() used not to find.
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
                            "recorded": sum(1 for label in self._pool
                                            for t in self.takes(label)
                                            if t.recorded),
                            "model": self.model}
        self._prune()
        log.info("face gallery saved: generation %d, %d samples (%s)", gen, n, reason)
        return gen

    def _sample_count(self, generation: int) -> int:
        """How many usable embeddings one generation holds, without loading it.

        Provenance already carries ``n``; this reads the file for it rather
        than trusting a number in memory, because the whole point of the two
        callers is to defend against a caller whose memory is empty."""
        try:
            _pool, _takes, prov = self._read(self.path_for(generation))
        except Exception:
            return 0          # unreadable: it defends nothing and protects nothing
        if str(prov.get("model") or LEGACY_MODEL) != self.model:
            # ANOTHER MODEL'S GENERATION DEFENDS NOTHING HERE, and counting it
            # would break the very first save after a swap: his 13 SFace takes
            # would be the shrink guard's baseline, and a fresh 5-take ArcFace
            # enrolment would be refused as "shrinking the gallery from 13 to
            # 5". It is still protected from pruning -- see ``_prune`` -- it
            # simply is not evidence about THIS model's enrolment.
            return 0
        return int(prov.get("n") or 0)

    def _on_disk_n(self) -> int:
        """The newest generation that parses, in samples. 0 if there is none.

        THIS IS THE FIX FOR THE INCIDENT'S OWN SHAPE. ``enroll_from_audio``
        built a fresh verifier and saved; a fresh object has loaded nothing,
        so a purely in-memory baseline is 0 and the shrink guard abstains on
        exactly the case it was written for. Asking the disk costs one small
        npz read on a path that runs at most a dozen times a year."""
        for gen in reversed(self.generations()):
            n = self._sample_count(gen)
            if n:
                return n
        return 0

    def _check_not_collapsed(self) -> None:
        """Every sample of a label being the same vector is the state the
        incident left the voiceprint in, and it is indistinguishable from a
        working enrolment until the day it refuses him."""
        for label, pool in self._pool.items():
            if len(pool) < 2:
                continue
            if all(cosine(pool[0], e) > 0.9999 for e in pool[1:]):
                raise ValueError(
                    "refusing to save a collapsed pool for %r: all %d samples "
                    "are the same vector" % (label, len(pool)))

    def _prune(self) -> None:
        """Drop the oldest generations past the window -- but NEVER the richest.

        Oldest-first alone makes the generations self-destructing: five saves
        of any size at all evict a six-take enrolment, and the store lands in
        the state the incident left, one loop later. Measured 2026-09-02: one
        6-sample enrolment plus six 2-sample saves left [4,5,6,7,8], all n=2.

        Protecting ``max(n)`` costs nothing in the normal case -- when saves
        are the same size or growing, the richest IS the newest and is kept
        anyway, so this changes the outcome only when a bigger generation is
        about to fall out of the window, which is the only case that matters.
        Ties go to the newest, so a steady state prunes exactly as before."""
        gens = self.generations()
        keep = max(KEEP_GENERATIONS, MIN_GENERATIONS)
        # ANOTHER MODEL'S GENERATIONS ARE NOT IN THE WINDOW AT ALL. Without
        # this line the first five saves after a model swap would evict his
        # entire previous enrolment -- silently, as a side effect of a
        # successful re-enrolment, leaving him nothing to revert to. That is
        # the same shape as the incident this whole module exists for, one
        # model change later. They are pruned by ``purge()``, which is a
        # deliberate delete, and by nothing else.
        mine = self._own_generations(gens)
        kept_foreign = [g for g in gens if g not in mine]
        if kept_foreign:
            log.debug("face gallery: %d generation(s) from another model are "
                      "outside the prune window", len(kept_foreign))
        gens = mine
        for tmp in self._tmp_paths():
            # A crashed save's leftovers are not a generation and hold no
            # history worth keeping; they are just embeddings lying around --
            # which is why they are shredded rather than unlinked.
            try:
                _shred(tmp)
            except OSError:
                log.debug("could not remove %s", tmp.name, exc_info=True)
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

    def _own_generations(self, gens=None) -> List[int]:
        """The generations THIS gallery's model wrote, oldest first.

        An unreadable generation is counted as its own -- it may be one of
        ours and there is no way to know, and the alternative is a pruning
        rule that quietly protects corrupt files forever.
        """
        out = []
        for gen in (self.generations() if gens is None else list(gens)):
            try:
                _pool, _takes, prov = self._read(self.path_for(gen))
            except Exception:  # noqa: BLE001
                out.append(gen)
                continue
            if str(prov.get("model") or LEGACY_MODEL) == self.model:
                out.append(gen)
        return out

    def rollback(self) -> int:
        """Delete the newest generation and load the one before it.

        The step that did not exist on 2026-09-02. Returns the generation now
        loaded, or 0 if there was nothing to roll back to."""
        # OUR generations only: rolling back an ArcFace enrolment must never
        # shred the SFace file underneath it.
        gens = self._own_generations()
        if len(gens) < 2:
            return 0
        # Prove there is something to fall back TO before destroying what
        # is here. With gen 1 unreadable (a 0-byte truncation, a FORMAT
        # bump) and gen 2 the only good enrolment, this used to shred gen 2
        # on a file COUNT, fail the load, return 0 -- and the caller then
        # printed "nothing to roll back to" over an empty directory (F13,
        # reproduced 2026-09-03). An unreadable generation defends nothing.
        if not any(self._sample_count(g) > 0 for g in gens[:-1]):
            return 0
        _shred(self.path_for(gens[-1]))
        self.loaded_generation = 0
        self._loaded_n = 0
        # load() falls back down the stack when gens[-2] is ALSO corrupt, so
        # returning gens[-2] reports a generation this object may not be
        # holding. Measured 2026-09-02: with gen 2 corrupted, rollback()
        # returned 2 while loaded_generation was 1 -- an enrolment script
        # printing "rolled back to generation 2" over generation 1 is the
        # quiet mismatch this whole module exists to prevent.
        return self.loaded_generation if self.load() else 0

    def purge(self) -> int:
        """Delete every generation AND every crashed save's tmp; return files
        removed.

        "Deleted" has to mean deleted. Removing only the newest leaves an
        older measurement of his face on the disk, which answers the wrong
        question -- and so does removing only the files whose names match the
        generation pattern, because ``gen-00002.npz.tmp`` does not and holds
        the same embeddings.

        Every file is OVERWRITTEN before it is unlinked (``_shred``): unlink
        alone drops the directory entry and leaves the vectors in the extents.
        Read ``_shred`` for what that does and does not buy -- the command
        that calls this is only allowed to claim the part that is true."""
        removed = 0
        for path in [self.path_for(g) for g in self.generations()] + self._tmp_paths():
            try:
                _shred(path)
                removed += 1
            except OSError:
                log.warning("could not delete face gallery file %s", path.name,
                            exc_info=True)
        self._pool = {}
        self._takes = {}
        self.loaded_generation = 0
        self._loaded_n = 0
        self._provenance = {}
        log.info("face gallery purged: %d generations deleted", removed)
        return removed

    def purge_label(self, label: str, reason: str = "") -> dict:
        """Destroy ONE person's embeddings, everywhere on the disk.

        THE REASON THIS IS NOT ``forget()`` + ``save()``. A save writes a new
        generation; the older ones still hold her, one ``rollback()`` away
        and, more to the point, still lying on the disk as 128 floats per
        take. "Delete me" is the one promise in this module that a new
        generation cannot keep, because what was asked for is the absence of
        the data and not the absence of a match.

        So the order is the same as ``--reset``'s and for the same reason:
        WRITE WHAT IS LEFT FIRST, DESTROY SECOND. Everyone else's embeddings
        land in a fresh generation before a single old file is touched, and
        if that write fails nothing is destroyed at all -- a delete of one
        person may never cost another person's enrolment.

        ONLY A GENERATION PROVEN TO HOLD HER IS DESTROYED. An earlier version
        shredded every generation that would not PARSE as well, reasoning
        that nothing can prove an unreadable file does not hold her. That is
        true and it is not worth what it costs: ``_read`` REFUSES a format
        number it does not recognise (that is the point of the check), so the
        first build that bumps FORMAT makes every existing generation
        "unreadable", and ``--delete --label somebody-who-was-never-enrolled``
        would then destroy the whole gallery and exit 0. Measured 2026-09-03
        on a throwaway store: 13 embeddings, one generation, ``_format``
        bumped by one, ``--delete --label heather`` -> empty directory.
        Unreadable generations are now COUNTED AND REPORTED and never
        touched; the caller says so out loud and stops claiming the delete
        was complete. Destroying them is still available and still one
        command -- it is ``--delete`` with no ``--label``, which is the one
        that means "everything".

        CRASHED-SAVE LEFTOVERS GO, ALWAYS. ``gen-00002.npz.tmp`` holds a full
        pool and does not match ``_GEN_RE``, so it is invisible to
        ``generations()`` and to the caller's read-back -- and it cannot be
        filtered by label, because it is somebody's whole pool. ``_prune()``
        already shreds them on every ordinary save, which is exactly why this
        was invisible: when somebody else survives, the save cleans up on the
        way past. When NOBODY survives there is no save, and her complete
        embedding set stayed on the disk under a command that printed
        "verified". So they are shredded here, unconditionally, and counted.

        Every file goes through ``_shred`` -- overwritten, then unlinked.
        Read ``_shred`` for the limit of what that buys; the caller is only
        allowed to claim that part.

        Returns numbers, so a script can print them and a test can read them:
        which generations held her, how many files went, how many could not
        be read and so were LEFT, which generation holds what is left, and
        who is still enrolled.
        """
        if self.root is None:
            raise ValueError("this gallery has no root; it cannot be purged")
        label = str(label)
        out: dict = {"label": label, "generations_with": [], "removed": 0,
                     "unreadable": [], "tmp_removed": 0,
                     "generation": 0, "left": 0,
                     "labels_left": (), "reason": ""}
        holds: List[int] = []
        unreadable: List[int] = []
        for gen in self.generations():
            try:
                pool, _takes, _prov = self._read(self.path_for(gen))
            except Exception:
                unreadable.append(gen)
                continue
            if pool.get(label):
                holds.append(gen)
        out["generations_with"] = list(holds)
        out["unreadable"] = list(unreadable)
        if not holds:
            # Nothing on the disk was PROVEN to hold her, so nothing on the
            # disk may be destroyed on her account. The tmps still go: they
            # are a crashed save's leftovers, the next ordinary save would
            # shred them anyway, and one of them may be the very pool she is
            # asking to have removed.
            out["tmp_removed"] = self._shred_tmps()
            return out

        self.load()
        self.forget(label)
        if self.total():
            try:
                # allow_shrink: removing a person IS a shrink, and it is the
                # deliberate kind the guard exists to let through when it is
                # asked for by name.
                out["generation"] = self.save(
                    reason=reason or ("forget %s" % label), allow_shrink=True)
            except ValueError as exc:
                # The write that was going to carry everyone else forward
                # failed. Destroying the old generations now would take them
                # with her -- and the tmps stay too, because "nothing was
                # destroyed" has to mean nothing.
                out["reason"] = str(exc)
                log.warning("face gallery: not deleting %r -- what is left "
                            "could not be saved: %s", label, exc)
                return out
        # No survivors means no save, and that is fine HERE and only here:
        # the invariant the save protects is somebody else's enrolment, and
        # when the pool is empty there is nobody else to lose.
        for gen in holds:
            if gen == out["generation"]:
                continue
            path = self.path_for(gen)
            if not path.exists():
                continue          # _prune() may already have taken it
            try:
                _shred(path)
                out["removed"] += 1
            except OSError:
                log.warning("could not delete face gallery generation %d",
                            gen, exc_info=True)
        out["tmp_removed"] = self._shred_tmps()
        out["left"] = self.total()
        out["labels_left"] = self.labels()
        log.info("face gallery: %r removed from %d generation(s); %d "
                 "embeddings over %d label(s) left; %d generation(s) could "
                 "not be read and were LEFT ALONE",
                 label, out["removed"], out["left"], len(out["labels_left"]),
                 len(unreadable))
        return out

    def _shred_tmps(self) -> int:
        """Overwrite and unlink every ``gen-NNNNN.npz.tmp``; return the count.

        A tmp is a whole pool that no name can filter, invisible to
        ``generations()`` and therefore to any read-back that walks them.
        ``_prune()`` already treats them as garbage on every save."""
        gone = 0
        for tmp in self._tmp_paths():
            try:
                _shred(tmp)
                gone += 1
            except OSError:
                log.warning("could not delete face gallery leftover %s",
                            tmp.name, exc_info=True)
        return gone

    def drop_generations(self, generations) -> int:
        """Delete exactly these generations; return how many went.

        SEPARATE FROM ``purge()`` ON PURPOSE, and the difference is the whole
        point. ``purge()`` is "destroy everything and leave the object empty",
        which is only ever correct when the answer to "what will he have
        afterwards" is "nothing, and he asked for that". This is the last step
        of REPLACING: the caller has already written a new generation that it
        holds a number for, and only the ones that predate it are going. It
        never touches the in-memory pool or ``loaded_generation``, and a
        generation that is not on disk is skipped rather than guessed at.

        ``scripts/face_enrol.py --reset`` is the caller. It used to purge
        BEFORE the capture, so a run that then failed a check -- 'too tight' /
        'too loose', the outcome the whole design exists to produce -- left
        him with an empty directory and nothing to roll back to. Capture
        first, destroy last: the old generations survive every failure mode of
        the run, and go only once a new one is safely on disk.
        """
        wanted = {int(g) for g in generations}
        removed = 0
        for gen in self.generations():
            if gen not in wanted:
                continue
            try:
                _shred(self.path_for(gen))
                removed += 1
            except OSError:
                log.warning("could not delete face gallery generation %d",
                            gen, exc_info=True)
        if removed:
            log.info("face gallery: %d superseded generation(s) deleted",
                     removed)
        return removed

    def provenance(self) -> dict:
        return dict(self._provenance)


def default_gallery(model: Optional[str] = None,
                    backend: Optional[str] = None) -> FaceGallery:
    """The user's gallery, wherever PATHS says it is, for the live model.

    PATHS.FACE_GALLERY honours JARVIS_FACE_GALLERY, which tests/conftest.py
    forces into a throwaway directory -- so importing this in a test cannot
    reach his enrolled face.

    ``model`` wins if given; otherwise the ACTIVE BACKEND decides, which is
    what makes the swap arrive here without every call site being edited.
    ``facemodels`` is imported inside the function so this module still loads
    with nothing else present.
    """
    if model is None:
        from jarvis import facemodels as fm      # noqa: PLC0415 - keeps this
        model = fm.backend_for(backend).embed_model  # module dependency-free
    return FaceGallery(root=PATHS.FACE_GALLERY, model=model)
