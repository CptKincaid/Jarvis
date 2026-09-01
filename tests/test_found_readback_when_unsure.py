"""#25 "read-back when unsure — doesnt trigger, just ask to say that again".

The feature is real and was already built: `commander._confirm_or_run`
parks a timer/alarm/reminder behind a spoken read-back whenever
`Commander.shaky_transcript()` is true, i.e. the Whisper avg_logprob came
in under `confirm.shaky_logprob` (-0.7 in his assistant.json).

What he saw is the second half of his note, and it is exactly what the log
says. The transcript confidence gate ran in `app._process_audio` BEFORE the
commander saw a syllable, and it rejected at -0.85 — so the read-back could
only ever fire in the sliver between -0.85 and -0.70, and everything
shakier than that was answered with "Say that again, sir?" instead. Both
read-backs that DID reach him that evening were the OTHER trigger (a bulk
destructive, `_wants_read_back` with n > 1), and both of his answers were
then eaten by the same gate:

    20:58:21.098 speaking (f5): Clear all three off your shopping list, sir?
    20:58:31.612 Transcribed: 'Yes. Thank you.' (avg_logprob=-1.45)
    20:58:31.613 Rejected: confidence too low (-1.45 < -0.85)
    20:58:31.613 speaking (f5): Say that again, sir?

Nothing here is a new fix: the gate moved to -2.90 and grew
`_salvage_low_confidence` (tests/test_confidence_gate.py) after that
session. These tests are the missing end-to-end proof that the two halves
now meet — a shaky command reaching the read-back at all, and the read-back
surviving a shaky yes — because neither file tested the join, and every one
of his creation commands that evening happened to be confident (-0.27 to
-0.63), so the feature was never actually exercised.
"""
from __future__ import annotations

import time

import pytest

import jarvis.app as app_mod
from jarvis.config import CONFIG
from jarvis.transcriber import TranscribeResult
from tests.test_app_wiring import build, paths, seams  # noqa: F401  (fixtures)


@pytest.fixture
def app(build):          # noqa: F811 - the shared factory, one app per test
    return build()


def _result(text: str, conf: float) -> TranscribeResult:
    return TranscribeResult(text=text, confidence=conf,
                            segments=[(text, conf)])


def _speak(app, monkeypatch, text: str, conf: float) -> list[str]:
    """One clip through the whole voice path; returns what was SAID."""
    spoken: list[str] = []
    monkeypatch.setattr(CONFIG, "speaker_verify", False)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(app, "_say", lambda t, **kw: spoken.append(t))
    monkeypatch.setattr(app.transcriber, "transcribe",
                        lambda audio: _result(text, conf))
    app._process_audio(object())
    return spoken


# A timer he plainly said, transcribed correctly, scored badly — the shape
# the read-back exists for. -1.2 is inside the band the old gate ate.
def test_a_shaky_timer_is_read_back_not_answered_with_say_again(app, monkeypatch):
    spoken = _speak(app, monkeypatch, "set a timer for 10 minutes", -1.2)

    assert app_mod.SAY_AGAIN_LINE not in spoken, \
        "his #25 verbatim: the read-back did not trigger, it asked again"
    assert spoken, "the shaky command was swallowed in silence"
    question = spoken[-1]
    assert question.rstrip().endswith("?"), \
        f"expected the parse read back as a question, got {question!r}"
    assert "10 minutes" in question or "ten minutes" in question
    # ...and it is an OFFER: nothing is running until he says yes.
    assert app.commander._pending_destructive is not None


def test_a_confident_timer_stays_zero_friction(app, monkeypatch):
    """The read-back is the doubtful tail only. A clean transcript must
    not grow a confirmation step — every timer he set that evening scored
    between -0.27 and -0.63."""
    spoken = _speak(app, monkeypatch, "set a timer for 10 minutes", -0.4)
    assert not any(s.rstrip().endswith("?") for s in spoken), \
        f"a confident command was made to confirm itself: {spoken!r}"
    assert app.commander._pending_destructive is None


def test_the_yes_that_answers_the_read_back_survives_its_own_bad_score(
        app, monkeypatch):
    """The other half of the same evening: the read-back fired and then his
    answer was rejected for confidence (20:58:31, 'Yes. Thank you.' at
    -1.45). A read-back that cannot be answered is worse than none."""
    ran: list[str] = []
    app.commander._pending_destructive = (
        lambda: ran.append("set"), "A timer for ten minutes, sir?",
        time.monotonic())

    _speak(app, monkeypatch, "Yes. Thank you.", -1.45)
    assert ran == ["set"], "his own yes was thrown away by the gate again"
