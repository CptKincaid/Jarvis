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
# A crashed save leaves gen-00002.npz.tmp, which _GEN_RE does not match -- so
# the first purge() reported success and left a full set of his embeddings on
# disk under a name nothing looked for. Deletion has to mean deletion.
_TMP_RE = re.compile(r"^gen-(\d{5})\.npz\.tmp$")
_KEY_RE = re.compile(r"^emb_(.+)_(\d{4})$")
# A label goes into an npz key and a log line, so keep it boring.
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,30}$")


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

    def _tmp_paths(self) -> List[Path]:
        """Any ``gen-NNNNN.npz.tmp`` a crashed save left. These hold a full
        set of embeddings and are invisible to ``_GEN_RE``."""
        if self.root is None or not self.root.is_dir():
            return []
        return sorted(p for p in self.root.iterdir() if _TMP_RE.match(p.name))

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
            for i, emb in enumerate(pool):
                arrays["emb_%s_%04d" % (label, i)] = emb
        arrays["_format"] = np.array([FORMAT])
        arrays["_created_ns"] = np.array([time.time_ns()])
        arrays["_reason"] = np.array([str(reason)])

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
                            "reason": str(reason), "n": n}
        self._prune()
        log.info("face gallery saved: generation %d, %d samples (%s)", gen, n, reason)
        return gen

    def _sample_count(self, generation: int) -> int:
        """How many usable embeddings one generation holds, without loading it.

        Provenance already carries ``n``; this reads the file for it rather
        than trusting a number in memory, because the whole point of the two
        callers is to defend against a caller whose memory is empty."""
        try:
            _pool, prov = self._read(self.path_for(generation))
        except Exception:
            return 0          # unreadable: it defends nothing and protects nothing
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

    def rollback(self) -> int:
        """Delete the newest generation and load the one before it.

        The step that did not exist on 2026-09-02. Returns the generation now
        loaded, or 0 if there was nothing to roll back to."""
        gens = self.generations()
        if len(gens) < 2:
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
        self.loaded_generation = 0
        self._loaded_n = 0
        self._provenance = {}
        log.info("face gallery purged: %d generations deleted", removed)
        return removed

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


def default_gallery() -> FaceGallery:
    """The user's gallery, wherever PATHS says it is.

    PATHS.FACE_GALLERY honours JARVIS_FACE_GALLERY, which tests/conftest.py
    forces into a throwaway directory -- so importing this in a test cannot
    reach his enrolled face."""
    return FaceGallery(root=PATHS.FACE_GALLERY)
