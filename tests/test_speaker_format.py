"""The loader refuses a voiceprint format it does not know -- and the test
that measures what happens when it does not.

WHY THIS FILE SHIPS BEFORE THE MULTI-SPEAKER STORE EXISTS. ``speaker.load()``
picks its keys with ``k.startswith("emb_")``. That pattern matches
``emb_0000`` and it also matches ``emb_mara_0000``. The only format complaint
the loader ever had fired on ``fmt < VOICEPRINT_FORMAT`` -- a LOWER number --
so a file written by a later, multi-label build read back on this one as one
big single-speaker pool, silently. Every person in it becomes part of one
centroid, and everybody in the file is then accepted as the owner.

That is not a hypothetical: it is the shape a ROLLBACK takes, or a stale
process that has not been restarted since the store changed. It is also the
one failure the transcript gate exists to prevent (a television reaching the
commander), arriving by a path nobody would look at, so it is closed first and
on its own.

Every vector here is SYNTHETIC. Nothing in this file touches the microphone,
his enrolled voiceprint, or any recording.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import speaker as sp
from tests.synthvoice import Voices, cos as _cos


def _person(seed, n):
    """``n`` takes of one pretend speaker, at his measured within-person
    spread. See tests/synthvoice.py for what is calibrated and what is not."""
    return Voices(seed=seed).takes("someone", n)


def _write(path, arrays, fmt):
    payload = dict(arrays)
    payload["_format"] = np.array([fmt])
    with open(path, "wb") as fh:
        np.savez(fh, **payload)


@pytest.fixture()
def at(tmp_path, monkeypatch):
    """Point the module-global voiceprint path at a throwaway file.

    ``speaker.py`` reads ``PATHS.VOICEPRINT`` at IMPORT into
    ``VOICEPRINT_FILE``, so monkeypatching the PATHS attribute alone would
    change nothing -- the same trap tests/conftest.py's forced env var exists
    for. Patch the module global.
    """
    path = tmp_path / "voiceprint.npz"
    monkeypatch.setattr(sp, "VOICEPRINT_FILE", path)
    return path


# ---------------------------------------------------- the known formats load
@pytest.mark.parametrize("fmt", sp.KNOWN_VOICEPRINT_FORMATS)
def test_a_format_this_build_knows_still_loads(at, fmt):
    """Format 1 (pre-trim) and format 2 (his live one) both still load. The
    refusal below must not cost him the file he actually has."""
    pool = _person(1, 6)
    _write(at, {"emb_%04d" % i: e for i, e in enumerate(pool)}, fmt)
    v = sp.SpeakerVerifier()
    v.load()
    assert v.is_enrolled is True
    assert v.num_samples == 6
    assert v.format_fault == ""


def test_a_voiceprint_with_no_format_key_is_read_as_format_1(at):
    """The oldest files predate the key entirely. Absent has always meant 1
    and must keep meaning 1, or this refusal deletes his history."""
    pool = _person(2, 4)
    with open(at, "wb") as fh:
        np.savez(fh, **{"emb_%04d" % i: e for i, e in enumerate(pool)})
    v = sp.SpeakerVerifier()
    v.load()
    assert v.is_enrolled is True and v.num_samples == 4


# ------------------------------------------- the trap, measured, then closed
def test_a_multi_label_future_format_would_accept_a_stranger_as_him(at):
    """THE MEASUREMENT THIS WHOLE COMMIT IS FOR, done with the refusal
    switched off so the number is the real one.

    Two synthetic people, written the way a multi-label store would write
    them (``emb_<label>_<nnnn>``) under a format this build has never seen.
    With the old rule -- warn only when the number is LOWER -- every vector
    in the file joins one centroid, and the OTHER person's takes then clear
    the 0.30 accept bar as the owner.
    """
    world = Voices(seed=3)
    him, her = world.takes("hunter", 14), world.takes("mara", 6)
    arrays = {}
    for i, e in enumerate(him):
        arrays["emb_hunter_%04d" % i] = e
    for i, e in enumerate(her):
        arrays["emb_mara_%04d" % i] = e
    _write(at, arrays, 3)

    # What the OLD loader did: take every emb_ key, pool it, ask no questions.
    data = np.load(at)
    keys = sorted(k for k in data.files if k.startswith("emb_"))
    assert len(keys) == 20, "the old key pattern matches labelled keys too"
    pooled = np.mean([np.asarray(data[k], dtype=np.float64) for k in keys],
                     axis=0)
    passed = sum(1 for e in her if _cos(e, pooled) >= sp.DEFAULT_THRESHOLD)
    assert passed == len(her), (
        "the pooled centroid was expected to admit every one of her takes as "
        "him; %d of %d did" % (passed, len(her)))

    # And what this build does instead.
    v = sp.SpeakerVerifier()
    v.load()
    assert v.is_enrolled is False, "a format this build cannot read was loaded"
    assert v.num_samples == 0


def test_an_unknown_format_is_refused_by_number_and_says_so(at):
    _write(at, {"emb_hunter_%04d" % i: e
                for i, e in enumerate(_person(4, 5))}, 9)
    v = sp.SpeakerVerifier()
    v.load()
    assert v.is_enrolled is False
    assert "format 9" in v.format_fault
    # By NAME, not "an error occurred": the sentence has to be actionable in
    # a log he reads at 2am.
    assert "9" in v.format_fault and "2" in v.format_fault


def test_the_refusal_leaves_the_file_alone(at):
    """It is his rollback. Refusing to READ it must never be a reason to
    touch it -- facegallery makes the same promise about a foreign
    generation."""
    _write(at, {"emb_hunter_0000": _person(5, 1)[0]}, 7)
    before = at.read_bytes()
    sp.SpeakerVerifier().load()
    assert at.exists() and at.read_bytes() == before


def test_a_refused_format_fails_OPEN_on_both_gates(at):
    """The direction matters more than the refusal.

    With nothing loaded, ``is_enrolled`` is False, so ``verify`` and
    ``filter_segments`` take their nothing-enrolled branches -- both of which
    pass audio through. Voice keeps working, unfiltered and loudly logged.
    Nobody is NAMED from an unreadable pool, which is the part that closes
    the false accept; the audio itself is not thrown away, which is the part
    that keeps a fresh box from going mute.
    """
    _write(at, {"emb_hunter_%04d" % i: e
                for i, e in enumerate(_person(6, 8))}, 11)
    v = sp.SpeakerVerifier()
    v.load()
    audio = np.zeros(16000 * 4, dtype=np.float32)
    ok, score = v.verify(audio)
    assert ok is True and score == 1.0
    out, stats = v.filter_segments(audio)
    assert out is audio
    assert stats["matched"] == 0 and stats["total"] == 0


def test_the_fault_clears_when_a_readable_file_replaces_it(at):
    _write(at, {"emb_hunter_0000": _person(7, 1)[0]}, 12)
    v = sp.SpeakerVerifier()
    v.load()
    assert v.format_fault
    _write(at, {"emb_%04d" % i: e for i, e in enumerate(_person(8, 6))}, 2)
    v.load()
    assert v.format_fault == "" and v.num_samples == 6
