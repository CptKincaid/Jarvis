"""Synthetic speaker embeddings, calibrated against his real pool's NUMBERS.

THERE IS NO SECOND PERSON'S VOICE ON THIS BOX. That is the honest limit of
everything multi-speaker, and it is why this module exists: every
between-people test in the suite runs on vectors generated here, and none of
them may be read as a measurement of two real humans.

WHAT IS CALIBRATED, AND WHAT IS NOT.

Calibrated -- the SAME-PERSON numbers, against ~/.aiws_trainer/voiceprint.npz
read as floats on 2026-09-04 (14 takes, format 2, no audio touched):

    pairwise cosine   min 0.283  median 0.485  max 0.735
    sample vs the pool's own centroid            0.632 - 0.831
    leave-one-out (sample vs the other 13)       0.570 - 0.798

``speaker(...)`` reproduces those by construction: a take is
``c*shared + identity + s*noise`` with ``s`` solved from
``same = (c^2+1)/(c^2+1+s^2)``, so its within-person spread is his, not a
tidier one. A fixture tighter than the real pool would make every threshold in
these tests easier to pass than reality.

NOT calibrated -- the BETWEEN-people number. ``c`` is the size of the
component two speakers share, and nothing on this machine can measure it: the
"non-match" figures in speaker.py (-0.03..-0.10) are silence, a television and
room tone, not a second human. ``c`` is therefore a DIAL, set per test to
produce the situation the test is about (two people far apart, or two people
close enough to be confusable), and no test may quote a cross-person cosine
from here as evidence about Mara or Heather. It is evidence about the CODE's
behaviour at a given separation, which is all a test can ever be here.
"""
from __future__ import annotations

import numpy as np

# His pool's median pairwise cosine, measured. The default same-person
# tightness, so a fixture is his spread unless a test says otherwise.
HIS_MEDIAN_PAIRWISE = 0.485
# His stored embeddings' L2 norms run 242-355. ECAPA vectors are not unit
# length and speaker._cosine_similarity normalises at compare time; giving the
# fixtures a realistic magnitude keeps a "forgot to normalise" bug visible.
HIS_NORM = 300.0
DIM = 192


def _unit(rng, n=DIM):
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


class Voices:
    """A world of pretend speakers who all share one ``shared`` direction.

    ``apart`` is that shared component's size. It sets how confusable two
    people are and it is the one number here with no real-world referent --
    see the module docstring.
    """

    def __init__(self, seed=0, apart=0.7, same=HIS_MEDIAN_PAIRWISE,
                 norm=HIS_NORM, dim=DIM):
        self.rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.apart = float(apart)
        self.same = float(same)
        self.norm = float(norm)
        self.shared = _unit(self.rng, self.dim)
        self._identity = {}
        # s^2 from same = (c^2+1)/(c^2+1+s^2): the noise size that reproduces
        # his measured within-person spread at this separation.
        c2 = self.apart ** 2
        self._s = float(np.sqrt((c2 + 1.0) * (1.0 / self.same - 1.0)))

    def identity(self, who):
        """One speaker's own direction, orthogonal to the shared one so that
        ``apart`` alone controls the overlap between two people."""
        if who not in self._identity:
            v = _unit(self.rng, self.dim)
            v = v - self.shared * float(v @ self.shared)
            self._identity[who] = v / np.linalg.norm(v)
        return self._identity[who]

    def take(self, who):
        """One utterance by ``who``, as float32 at a realistic magnitude."""
        v = (self.apart * self.shared + self.identity(who)
             + self._s * _unit(self.rng, self.dim))
        return (v / np.linalg.norm(v) * self.norm).astype(np.float32)

    def takes(self, who, n):
        return [self.take(who) for _ in range(n)]


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def centroid(vectors):
    return np.mean([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
