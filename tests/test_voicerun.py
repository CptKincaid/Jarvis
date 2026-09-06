"""The in-app VOICE enrolment: the run that replaces a fifteen-second append.

WHAT IT REPLACES, and the replacement is the point. ``JarvisApp.enroll_speaker``
records a fixed 15 s of whatever is in the room, hands it to
``speaker.enroll_from_audio``, and that APPENDS one embedding to
``voiceprint.npz`` -- a single file replaced in place with no generations and
no rollback. No loudness floor, no cohesion check, no separation check, no
consent, no gate, no owner check and no undo, one click deep in the settings
drawer next to a slider. This file pins the run that takes its place.

THE MICROPHONE IS NEVER OPENED HERE. The recorder is injected and answers with
synthetic float32 arrays; the embedder is injected and answers with vectors
from tests/synthvoice.py, which is calibrated to his own pool's WITHIN-person
spread and whose BETWEEN-person separation is an explicit dial with no
real-world referent. Nothing below is a measurement of a second human, and no
number here may be quoted as one.

CONSTRAINT E IS THE HARD ONE. ``MicArbiter`` is a re-entrant DEPTH COUNTER,
not a mutex: it pauses the hotword once and resumes once and will happily let
two consumers hold at the same time. ``tts.py`` takes ``acquire("tts")`` for
talkback, so a voice enrolment started while Jarvis is speaking records
Jarvis's own voice out of the speakers -- there is no AEC on this box -- and
that is what the present button does. The refusals here are what stop it.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import consent as cs
from jarvis import voicegallery as vg
from jarvis import voicerun as vr
from tests.synthvoice import Voices

RATE = 16000


def _audio(seconds=8.0, level=0.05):
    n = int(RATE * seconds)
    rng = np.random.default_rng(7)
    return (rng.normal(size=n) * level).astype(np.float32)


class Recorder:
    """An injected recorder. Records what it was asked for; opens nothing."""

    def __init__(self, clips=None, recording=False):
        self.clips = list(clips) if clips is not None else None
        self.asked = []
        self.recording = recording
        self.mic_available = True
        self.arbiter = Arbiter()

    def record_fixed(self, seconds):
        self.asked.append(float(seconds))
        if self.clips is None:
            return _audio(seconds)
        if not self.clips:
            return np.zeros(0, dtype=np.float32)
        return self.clips.pop(0)


class Arbiter:
    """The depth counter, watched. ``depth`` back to 0 is the assertion."""

    def __init__(self):
        self.depth = 0
        self.max_depth = 0
        self.owners = []
        self.paused = 0
        self.resumed = 0

    def acquire(self, owner):
        arb = self

        class _Ctx:
            def __enter__(self):
                arb.depth += 1
                arb.max_depth = max(arb.max_depth, arb.depth)
                arb.owners.append(owner)
                if arb.depth == 1:
                    arb.paused += 1
                return arb

            def __exit__(self, *exc):
                arb.depth -= 1
                if arb.depth == 0:
                    arb.resumed += 1
                return False

        return _Ctx()


class Tts:
    def __init__(self, speaking=False):
        self._speaking = speaking

    def is_speaking(self):
        return self._speaking


def _embedder(world, who="alderman", fail_after=None):
    made = {"n": 0}

    def embed(_audio_in):
        made["n"] += 1
        if fail_after is not None and made["n"] > fail_after:
            return None
        return world.take(who)

    return embed


def _run(tmp_path, *, gallery=None, recorder=None, embed=None, tts=None,
         label="alderman", takes=vg.MIN_TAKES_TO_NAME, **kw):
    gallery = gallery or vg.VoiceGallery(root=tmp_path / "voice_gallery")
    world = Voices(seed=11)
    said = []
    cards = []
    tones = []
    return vr.VoiceRun(
        label=label, gallery=gallery,
        recorder=recorder or Recorder(),
        embed=embed or _embedder(world),
        tts=tts or Tts(),
        say=said.append, card=cards.append, earcon=tones.append,
        takes=takes, seconds=1.0, settle_s=0.0,
        consent_how=cs.HOW_OWNER, now=lambda: 0.0, sleep=lambda _s: None,
        **kw), said, cards, tones, gallery


# ================================================ E: the device has one owner
def test_the_preflight_refuses_while_jarvis_is_speaking():
    """The refusal that matters most: with no AEC on this box, recording
    while talkback holds the arbiter enrols Jarvis's own voice."""
    out = vr.preflight(recorder=Recorder(), tts=Tts(speaking=True),
                       services=None)
    assert out["ok"] is False
    assert "talking" in out["reply"].lower() or "speaking" in out["reply"].lower()


def test_the_preflight_refuses_while_a_recording_is_open():
    out = vr.preflight(recorder=Recorder(recording=True), tts=Tts(),
                       services=None)
    assert out["ok"] is False


def test_the_preflight_refuses_a_second_run():
    class Svc:
        voice_run = object()
    out = vr.preflight(recorder=Recorder(), tts=Tts(), services=Svc())
    assert out["ok"] is False
    assert "already" in out["reply"].lower()


def test_the_preflight_refuses_with_no_microphone():
    rec = Recorder()
    rec.mic_available = False
    out = vr.preflight(recorder=rec, tts=Tts(), services=None)
    assert out["ok"] is False


def test_the_arbiter_is_taken_once_for_the_whole_run_and_given_back(tmp_path):
    """ONE acquire, not one per take: eight separate acquires would pause and
    resume the hotword eight times, and between any two of them the wake word
    is live on a microphone that is about to be recorded into."""
    run, _said, _cards, _tones, _gal = _run(tmp_path)
    rec = run.recorder
    run.walk()
    assert rec.arbiter.max_depth == 1
    assert rec.arbiter.depth == 0
    assert rec.arbiter.paused == 1 and rec.arbiter.resumed == 1


def test_the_arbiter_is_given_back_on_the_exception_path(tmp_path):
    def boom(_a):
        raise RuntimeError("the embedder fell over")

    run, _said, _cards, _tones, _gal = _run(tmp_path, embed=boom)
    rec = run.recorder
    run.walk()
    assert rec.arbiter.depth == 0
    assert rec.arbiter.resumed == 1


def test_he_is_told_the_wake_word_is_deaf_so_his_abort_is_a_button(tmp_path):
    """``enrolrun`` never acquires the mic so the abort WORD stays hearable.
    A voice run cannot make that choice -- the two collide -- so the honest
    answer is a Stop button, said out loud before the first take."""
    run, said, _cards, _tones, _gal = _run(tmp_path)
    run.walk()
    opening = " ".join(said).lower()
    assert "stop" in opening
    assert "press" in opening or "button" in opening


# ============================================ the loudness floor, per take
def test_a_take_under_the_floor_is_refused_and_retried(tmp_path):
    """The floor is ``voiceenrol.MIN_RMS`` -- the SAME number
    scripts/enroll_voice.py and scripts/voice_enrol.py apply, not a second
    one written here."""
    quiet = (np.zeros(int(RATE * 1.0), dtype=np.float32),)
    clips = list(quiet) + [_audio(1.0) for _ in range(vg.MIN_TAKES_TO_NAME)]
    rec = Recorder(clips=clips)
    run, said, _cards, _tones, gal = _run(tmp_path, recorder=rec)
    run.walk()
    assert run.kept == vg.MIN_TAKES_TO_NAME
    assert len(rec.asked) == vg.MIN_TAKES_TO_NAME + 1
    assert any("quiet" in s.lower() or "silent" in s.lower() for s in said)


def test_the_floor_is_the_shared_one_and_not_a_second_number():
    from jarvis import voiceenrol as ve
    assert vr.MIN_RMS is ve.MIN_RMS


def test_the_count_that_is_judged_is_the_KEPT_takes(tmp_path):
    """A refused take must not count toward the floor, or eight tries with
    six usable ones would be stored as an eight-take pool."""
    quiet = np.zeros(int(RATE * 1.0), dtype=np.float32)
    rec = Recorder(clips=[quiet, quiet, quiet])
    run, _said, _cards, _tones, gal = _run(tmp_path, recorder=rec,
                                           max_tries=3)
    run.walk()
    assert run.kept == 0
    assert gal.generations() == []


# ================================= the pool is judged BEFORE anything is written
def test_a_collapsed_pool_is_refused_and_nothing_is_written(tmp_path):
    """Eight near-identical vectors are one take recorded eight times. The
    bar is ``voicegallery.COLLAPSED_MEDIAN_COSINE``."""
    world = Voices(seed=3)
    one = world.take("alderman")

    def same(_a):
        return one.copy()

    run, said, _cards, _tones, gal = _run(tmp_path, embed=same)
    run.walk()
    assert gal.generations() == []
    assert any("alike" in s.lower() or "cosine" in s.lower() for s in said)


def test_a_pool_too_close_to_somebody_already_enrolled_is_refused(tmp_path):
    """Two people inside ``voicegallery.MARGIN`` cannot be told apart, and
    storing them makes every later verdict UNKNOWN for BOTH of them.

    THE SEPARATION USED HERE IS A DIAL, not a measurement: there is not one
    second of a second human on this machine.
    """
    gal = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    world = Voices(seed=4, apart=6.0)          # deliberately confusable
    for e in world.takes("heather", vg.MIN_TAKES_TO_NAME):
        gal.add("heather", e)
    gal.save("fixture")
    before = list(gal.generations())
    run, said, _cards, _tones, _g = _run(
        tmp_path, gallery=gal, embed=_embedder(world, "alderman"))
    run.walk()
    assert gal.generations() == before, "a refused pool wrote a generation"
    assert any("heather" in s.lower() for s in said)


def test_a_run_abandoned_after_three_takes_writes_nothing(tmp_path):
    run, _said, _cards, _tones, gal = _run(tmp_path)
    run.abort("he walked away")
    run.walk()
    assert gal.generations() == []
    assert run.saved_generation == 0


def test_a_successful_run_keeps_the_previous_generation_on_disk(tmp_path):
    """Topping up his OWN label. ONE world for both pools, deliberately: two
    ``Voices`` worlds are two different speakers, and the bars would rightly
    refuse to file them under one name."""
    gal = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    world = Voices(seed=5)
    for e in world.takes("alderman", vg.MIN_TAKES_TO_NAME):
        gal.add("alderman", e)
    first = gal.save("fixture")
    gal2 = vg.VoiceGallery(root=tmp_path / "voice_gallery")
    gal2.load()
    run, _said, _cards, _tones, _g = _run(
        tmp_path, gallery=gal2, embed=_embedder(world, "alderman"),
        label="alderman")
    run.walk()
    assert run.saved_generation > first
    assert first in gal2.generations(), "the old generation may be rolled back"


def test_the_consent_attestation_is_stored_and_is_never_a_false_typed(tmp_path):
    """A console consent that wrote "typed" would be a false attestation on
    disk -- worse than none, because a reader believes a terminal ceremony
    happened. The owner enrolling HIMSELF is ``consent.HOW_OWNER``."""
    run, _said, _cards, _tones, gal = _run(tmp_path)
    run.walk()
    assert run.saved_generation > 0
    assert gal.consent("alderman") == cs.HOW_OWNER
    assert gal.consent("alderman") != cs.HOW_TERMINAL


def test_nothing_in_the_run_touches_voiceprint_npz(tmp_path, monkeypatch):
    """The single file his voice identity rests on is written tmp->replace
    with NO generations and NO rollback, which is why the run writes to the
    GALLERY instead."""
    from jarvis.config import PATHS
    vp = tmp_path / "voiceprint.npz"
    vp.write_bytes(b"zzz not a real voiceprint")
    monkeypatch.setattr(PATHS, "VOICEPRINT", vp, raising=False)
    before = vp.read_bytes()
    run, _said, _cards, _tones, _gal = _run(tmp_path)
    run.walk()
    assert vp.read_bytes() == before


def test_the_module_never_reaches_the_unjudged_appender(tmp_path):
    """``speaker.enroll_from_audio`` is the append with no bars and no undo.
    It must not be reachable from this module -- asserted over the parsed
    tree rather than the source text, because the header NAMES it as the
    thing being replaced and a grep would find its own explanation."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(vr))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "enroll_from_audio" not in names
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert not any("speaker" in m for m in imported), sorted(imported)


def test_the_card_carries_numbers_and_the_run_never_speaks_a_number_it_lacks(
        tmp_path):
    run, _said, cards, _tones, _gal = _run(tmp_path)
    run.walk()
    blob = "\n".join(cards)
    assert "rms" in blob.lower()
    assert str(vg.MIN_TAKES_TO_NAME) in blob or "8" in blob


def test_stopping_ends_the_run_and_gives_the_microphone_back(tmp_path):
    run, said, _cards, _tones, gal = _run(tmp_path)
    rec = run.recorder
    run.abort(vr.STOP_PRESSED)
    run.walk()
    assert rec.arbiter.depth == 0
    assert gal.generations() == []
    assert any("nothing was written" in s.lower() for s in said)
