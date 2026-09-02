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
2. **A degenerate-vector guard.** The fixture was a constant vector. A real
   embedding never is (measured on this box 2026-09-02: cv2.FaceRecognizerSF
   returns 128 float32 with L2 = 10.41 and per-element spread), so the exact
   shape of that accident is refused at ``add()``.
3. **A collapsed-pool guard.** Two copies of one vector is not an enrolment,
   and is what the incident LEFT BEHIND. ``save()`` refuses it.
4. **A shrink guard.** Six samples became two and nothing objected. Dropping
   samples now needs ``allow_shrink=True``, which the enrolment script says
   out loud and a stray test does not.

And, belt and braces, the store obeys the test firewall the same way the
voiceprint now does: ``PATHS.FACE_GALLERY`` reads ``JARVIS_FACE_GALLERY``,
which tests/conftest.py forces into a throwaway directory before any jarvis
import. A firewall and a recoverable format are different defences and this
data warrants both -- the voiceprint had neither on the day it was lost.

FORMAT. One ``.npz`` per generation: ``emb_<label>_<nnnn>`` float32 arrays,
plus ``_format``, ``_created_ns`` and ``_reason``. Labels live in the key
because a face gallery holds more than one person the moment a second person
sits down -- distinguishing "someone" from "him" is the point.

DELETION is ``purge()``: every generation, not just the newest. A store whose
"delete" leaves an older copy of his face on disk has not deleted anything.

Nothing here imports cv2, torch or a model. It is arithmetic over arrays, so
it runs in the test suite with no camera, no display and no GPU.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from jarvis.config import PATHS
from jarvis.logs import get_logger

log = get_logger("facegallery")

FORMAT = 1
# SFace (face_recognition_sface_2021dec.onnx, Apache-2.0) returns 128 float32.
# Measured on this box 2026-09-02: shape (1, 128), L2 norm 10.41 -- NOT unit
# length, which is why every comparison here normalises rather than dotting.
EMBED_DIM = 128
# Five is enough history to undo a mistake noticed a few enrolments later and
# small enough that the whole store stays under a megabyte at 128 floats a
# sample. Never pruned below two: one generation is no backup at all.
KEEP_GENERATIONS = 5
MIN_GENERATIONS = 2
# OpenCV's documented SFace cosine threshold for "same person" is 0.363; this
# store does not enforce it (the consumer decides how sure it needs to be) but
# it is the number a caller should start from.
SFACE_COSINE_SAME = 0.363

_GEN_RE = re.compile(r"^gen-(\d{5})\.npz$")
_KEY_RE = re.compile(r"^emb_(.+)_(\d{4})$")
# A label goes into an npz key and a log line, so keep it boring.
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")


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


def degenerate_reason(vec) -> str:
    """"" if this could be a face embedding, else why it could not be.

    The three shapes that have actually appeared: the wrong dimension (a
    different model's output), a non-finite element (a crop of nothing), and a
    CONSTANT vector -- which is what the fixture that destroyed his voiceprint
    contained, and which no real embedding is."""
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.size != EMBED_DIM:
        return "wrong dimension: %d, expected %d" % (arr.size, EMBED_DIM)
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


class FaceGallery:
    """Enrolled faces, held in memory, persisted as numbered generations.

    ``root=None`` is a live, unsaveable gallery: useful for a caller that only
    wants to match against something it just built, and used by the tests that
    check the guards without touching a filesystem at all."""

    def __init__(self, root: Optional[Path] = None):
        self.root: Optional[Path] = None if root is None else Path(root)
        self._pool: Dict[str, List[np.ndarray]] = {}
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

    def reset(self) -> None:
        """Empty the in-memory pool. Touches no disk: the generations stay,
        which is what makes "enrol again from scratch" a safe thing to do."""
        self._pool = {}

    def add(self, label: str, vec) -> None:
        """Add one embedding, or raise ValueError saying why not."""
        if not _LABEL_RE.match(label or ""):
            raise ValueError("bad label %r: lowercase letters, digits, - and _" % label)
        why = degenerate_reason(vec)
        if why:
            raise ValueError("refusing a degenerate embedding: %s" % why)
        arr = np.asarray(vec, dtype=np.float32).ravel().copy()
        self._pool.setdefault(label, []).append(arr)

    def forget(self, label: str) -> int:
        """Drop one person from the in-memory pool; save() commits it."""
        gone = len(self._pool.pop(label, ()))
        return gone

    def match(self, vec) -> Tuple[str, float]:
        """The nearest enrolled label and its cosine, or ("", 0.0).

        Nearest by the pool's BEST sample rather than by a centroid: a
        centroid of him in glasses and him without is a face that does not
        exist, and the same averaging is why the voiceprint's own pool keeps
        its members (jarvis/speaker.py:249-262)."""
        if degenerate_reason(vec):
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

    def load(self, generation: Optional[int] = None) -> bool:
        """Load one generation, defaulting to the newest that parses.

        Falling back down the stack is the whole reason the stack exists: a
        truncated or corrupt newest file must cost the last enrolment, not the
        enrolment."""
        wanted = [generation] if generation else list(reversed(self.generations()))
        for gen in wanted:
            try:
                pool, prov = self._read(self.path_for(gen))
            except Exception:
                log.warning("face gallery generation %d unreadable; "
                            "falling back to the one before", gen, exc_info=True)
                continue
            self._pool = pool
            self.loaded_generation = gen
            self._loaded_n = sum(len(v) for v in pool.values())
            self._provenance = prov
            log.info("face gallery loaded: generation %d, %d samples over %d "
                     "labels", gen, self._loaded_n, len(pool))
            return True
        return False

    def _read(self, path: Path):
        data = np.load(path)
        fmt = int(data["_format"][0]) if "_format" in data.files else 0
        if fmt != FORMAT:
            # Reading a newer pool as if it were this one is how embeddings
            # silently stop comparing; speaker.py:264-271 warns about exactly
            # this for the voiceprint. Refuse, do not guess.
            raise ValueError("face gallery format %d, this build reads %d" % (fmt, FORMAT))
        pool: Dict[str, List[np.ndarray]] = {}
        for key in sorted(data.files):
            m = _KEY_RE.match(key)
            if not m:
                continue
            arr = np.asarray(data[key], dtype=np.float32).ravel()
            if degenerate_reason(arr):
                # A stored vector that cannot be a face is not loaded: it
                # would drag every future match toward itself.
                log.warning("face gallery: dropping %s (%s)", key, degenerate_reason(arr))
                continue
            pool.setdefault(m.group(1), []).append(arr)
        prov = {"format": fmt,
                "created_ns": int(data["_created_ns"][0]) if "_created_ns" in data.files else 0,
                "reason": str(data["_reason"][0]) if "_reason" in data.files else "",
                "n": sum(len(v) for v in pool.values())}
        return pool, prov

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
        if not allow_shrink and self._loaded_n and n < self._loaded_n:
            raise ValueError(
                "refusing to shrink the gallery from %d samples to %d; pass "
                "allow_shrink=True if you mean it (this guard exists because "
                "the voiceprint went 6 -> 2 unnoticed on 2026-09-02)"
                % (self._loaded_n, n))

        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        gen = (self.generations() or [0])[-1] + 1
        arrays: Dict[str, np.ndarray] = {}
        for label, pool in self._pool.items():
            for i, emb in enumerate(pool):
                arrays["emb_%s_%04d" % (label, i)] = emb
        arrays["_format"] = np.array([FORMAT])
        arrays["_created_ns"] = np.array([time.time_ns()])
        arrays["_reason"] = np.array([str(reason)])

        path = self.path_for(gen)
        tmp = path.with_name(path.name + ".tmp")
        # savez appends ".npz" to a bare path, so hand it a file handle and
        # keep the exact tmp name -- the same trap speaker.py:288-292 hit.
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        self.loaded_generation = gen
        self._loaded_n = n
        self._provenance = {"format": FORMAT, "created_ns": time.time_ns(),
                            "reason": str(reason), "n": n}
        self._prune()
        log.info("face gallery saved: generation %d, %d samples (%s)", gen, n, reason)
        return gen

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
        gens = self.generations()
        keep = max(KEEP_GENERATIONS, MIN_GENERATIONS)
        for gen in gens[:-keep] if len(gens) > keep else []:
            try:
                self.path_for(gen).unlink()
            except OSError:
                log.debug("could not prune generation %d", gen, exc_info=True)

    def rollback(self) -> int:
        """Delete the newest generation and load the one before it.

        The step that did not exist on 2026-09-02. Returns the generation now
        loaded, or 0 if there was nothing to roll back to."""
        gens = self.generations()
        if len(gens) < 2:
            return 0
        self.path_for(gens[-1]).unlink()
        self.loaded_generation = 0
        self._loaded_n = 0
        return gens[-2] if self.load() else 0

    def purge(self) -> int:
        """Delete EVERY generation and empty the pool; return files removed.

        "Deleted" has to mean deleted. Removing only the newest leaves an
        older measurement of his face on the disk, which answers the wrong
        question."""
        removed = 0
        for gen in self.generations():
            try:
                self.path_for(gen).unlink()
                removed += 1
            except OSError:
                log.warning("could not delete face gallery generation %d", gen,
                            exc_info=True)
        self._pool = {}
        self.loaded_generation = 0
        self._loaded_n = 0
        self._provenance = {}
        log.info("face gallery purged: %d generations deleted", removed)
        return removed

    def provenance(self) -> dict:
        return dict(self._provenance)


def default_gallery() -> FaceGallery:
    """The user's gallery, wherever PATHS says it is.

    PATHS.FACE_GALLERY honours JARVIS_FACE_GALLERY, which tests/conftest.py
    forces into a throwaway directory -- so importing this in a test cannot
    reach his enrolled face."""
    return FaceGallery(root=PATHS.FACE_GALLERY)
