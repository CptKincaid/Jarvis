"""THE SPOKEN PHRASE, AND EVERY LOG LINE THE TURN ACTUALLY WRITES.

The branch's own tests could not see this hole, and it is worth saying why
before the tests below are read. tests/test_knightfall_app.py stubs
``_decode_clip`` out entirely and stands in a transcriber that logs
nothing, so the ONE line that carries the words -- ``Transcribed: %r`` at
INFO in jarvis/transcriber.py, written before the gate has seen a syllable
-- was never executed by a test at all. The earlier fix closed that line on
the RESCUE leg only (``transcribe_quiet``, for a clip the speaker filter
dropped). The leg he is actually on is the other one: the speaker filter
MATCHES him, ``_decode_clip`` takes the loud ``transcribe``, and the phrase
lands in jarvis.log in plaintext (verdict, 2026-09-05, measured twice).

So these tests drive the REAL ``_decode_clip`` and the REAL
``_process_audio`` over the REAL ``Transcriber`` with a fake faster-whisper
model underneath, and assert on the LOG RECORDS OF EVERY LOGGER -- message,
args and formatted text -- rather than on a stand-in's call counts. Both
speaker-filter outcomes, all three modes.

No real phrase is anywhere near this file: the secret under test is
tests/test_owner_gate_wiring.FAKE_PHRASE, which is x's.
"""
from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import jarvis.app as app_mod
from jarvis import gate as gate_mod
from jarvis.config import CONFIG, PATHS
from jarvis.events import PartialText, Transcribed, UserUtterance, bus
from jarvis.transcriber import SAMPLE_RATE, Transcriber
from tests.test_decode_bounds import FakeFasterWhisper
from tests.test_owner_gate_wiring import FAKE_PHRASE, MATCHED, _stand_in

# What Whisper hands back for the clip in which he says it. Normalisation
# (jarvis/passphrase.normalise_spoken) folds the punctuation and the case,
# which is what makes this the same secret as FAKE_PHRASE.
SPOKEN = "Xxx, not a real phrase xxx."
# Every rendering of the secret a log line could plausibly carry.
SECRETS = (SPOKEN, SPOKEN.lower(), FAKE_PHRASE, "not a real phrase")

REFUSED = {"total": 2, "matched": 0, "scores": [0.11, 0.09]}


class _Speaker:
    """The speaker filter, as ``_decode_clip`` uses it. ``matches`` False is
    the clip it drops -- the rescue leg; True is the leg he is on."""

    enrolled = True

    def __init__(self, matches: bool):
        self.matches = matches

    def filter_segments(self, audio):
        if self.matches:
            return audio, dict(MATCHED)
        return None, dict(REFUSED)


def _real_transcriber(text, monkeypatch, tmp_path):
    """The REAL Transcriber over a fake faster-whisper model: the CPU
    branch, so no torch and no CUDA context are needed inside the suite."""
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    tr = Transcriber()
    tr._model = FakeFasterWhisper(segments=[
        SimpleNamespace(text=text, avg_logprob=-0.30, compression_ratio=1.2)])
    tr._gpu, tr._backend = False, "CPU int8"
    return tr


def _app(tmp_path, monkeypatch, *, mode, matches, said=SPOKEN, phrase=True):
    monkeypatch.setattr(CONFIG, "speaker_verify", True)
    a = _stand_in(tmp_path, mode=mode, phrase=phrase, said=said)
    a.transcriber = _real_transcriber(said, monkeypatch, tmp_path)
    a.speaker = _Speaker(matches)
    a.dispatched, a.nudges = [], []
    # Bound DEFENSIVELY, and that is not tidiness. A test that raises
    # AttributeError against the tip it is accusing proves nothing about the
    # tip; it has to RUN there and fail on the leak itself. The redaction
    # helpers below did not exist when this was first measured RED.
    for name in ("_process_audio", "_decode_clip", "_clip_decode",
                 "_log_transcript", "_gate_judge", "_gate_consumed",
                 "_gate_admits", "_drop_partial"):
        fn = getattr(app_mod.JarvisApp, name, None)
        if fn is not None:
            setattr(a, name, fn.__get__(a))
    a._take_speculation = lambda: None
    a._audio_busy = threading.Event()
    a._audio_busy.set()
    a._nudge = a.nudges.append
    a._maybe_learn_voice = lambda audio, stats: None
    a._dispatch = lambda text, source, **k: a.dispatched.append((text, source))
    a._salvage_low_confidence = lambda text, conf: ""
    a._say_again_count = 0
    a._stop_event = SimpleNamespace(t=0.0)
    return a


def _audio():
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


def _leaks(records):
    """Every record whose message, args or formatted text carries the
    secret. Checked three ways because a %r arg never reaches
    ``caplog.text`` until something formats it, and the file handler does."""
    out = []
    for r in records:
        blob = " ".join((str(r.msg), repr(r.args), r.getMessage())).lower()
        if any(s.lower() in blob for s in SECRETS):
            out.append(r)
    return out


@pytest.fixture
def events():
    seen = []
    subs = [(Transcribed, seen.append), (UserUtterance, seen.append),
            (PartialText, seen.append)]
    for etype, fn in subs:
        bus.subscribe(etype, fn)
    yield seen
    for etype, fn in subs:
        bus.unsubscribe(etype, fn)


# ====================================================== the log, measured
@pytest.mark.parametrize("mode", ["shadow", "enforce", "off"])
@pytest.mark.parametrize("matches", [True, False],
                         ids=["speaker-matches-him", "speaker-drops-the-clip"])
def test_the_consumed_phrase_turn_writes_no_words_to_any_log(
        tmp_path, monkeypatch, caplog, events, mode, matches):
    """THE WHOLE CLAIM, ON THE PATH HE IS ACTUALLY ON. Every logger, every
    level, both speaker outcomes, all three modes: zero records carry the
    words. The turn is still consumed and still dispatches nothing."""
    a = _app(tmp_path, monkeypatch, mode=mode, matches=matches)
    with caplog.at_level(logging.DEBUG):
        a._process_audio(_audio())
    leaked = _leaks(caplog.records)
    assert leaked == [], "leaking log records: %s" % (
        [(r.name, r.levelname, str(r.msg)) for r in leaked],)
    assert a.dispatched == []
    # Off says a different line, deliberately: it opens no window, so it
    # does not get to promise one (tests/test_knightfall_promise.py).
    expect = (gate_mod.PHRASE_OFF_LINE if mode == "off"
              else gate_mod.PHRASE_OK_LINE)
    assert a.spoken and a.spoken[0] == expect
    for ev in events:
        assert not _leaks([logging.LogRecord(
            "e", 20, "", 0, "%s", (str(vars(ev)),), None)])


def test_the_loud_decode_still_writes_an_ordinary_sentence_down(
        tmp_path, monkeypatch, caplog):
    """The redaction must cost the log nothing on an ordinary turn: the
    `Transcribed:` line is how a wrong answer gets diagnosed."""
    said = "what time is it"
    a = _app(tmp_path, monkeypatch, mode="shadow", matches=True, said=said)
    with caplog.at_level(logging.DEBUG):
        a._process_audio(_audio())
    assert a.dispatched == [(said, "voice")]
    assert any("Transcribed" in str(r.msg) and said in r.getMessage()
               for r in caplog.records), (
        "an ordinary turn's words must still be in the log")


def test_the_numbers_survive_the_redaction(tmp_path, monkeypatch, caplog):
    """avg_logprob is what the line is READ for; only the words go."""
    a = _app(tmp_path, monkeypatch, mode="shadow", matches=True)
    with caplog.at_level(logging.DEBUG):
        a._process_audio(_audio())
    assert any("avg_logprob=-0.30" in r.getMessage() for r in caplog.records)


def test_an_owner_with_no_phrase_set_pays_nothing(tmp_path, monkeypatch,
                                                  caplog):
    """No phrase enrolled, no secret to protect: the decode is the loud one
    and the log is exactly what it was before this branch existed."""
    said = "turn the kitchen lights off"
    a = _app(tmp_path, monkeypatch, mode="shadow", matches=True, said=said,
             phrase=False)
    with caplog.at_level(logging.DEBUG):
        a._process_audio(_audio())
    assert a.dispatched == [(said, "voice")]
    assert any("Transcribed" in str(r.msg) and said in r.getMessage()
               for r in caplog.records)
