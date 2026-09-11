"""A stop word stops the model job, not only the speech.

HIS BUG, 2026-09-11, in his own words: "locked me out by keep saying one
moment", and "when he was asked to look into [it] and claude was working
he refused to listen to voice commands and said one moment working. i had
to type cancel to stop him."

THE MECHANISM, measured on the tree before the fix. ``_h_quiet`` called
``_cut_speech`` (silencing the reply in flight) and ``claude.cancel()``
(stopping the Claude task) and NOTHING touched the brain, whose
``_acquire_busy`` guard then answered every utterance "Still on the last
one, sir. One moment." until ``BUSY_MAX_S`` (180 s) expired -- and spoke
the abandoned answer when the job landed. ``brain.cancel`` was reachable
from exactly one place in the whole commander: ``_try_correction``.

It is the rule the filled-pause lane wrote at eighteen grammar anchors
and this guard was never brought under: bar where ACTING is irreversible,
never where NOT acting is.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis import commander as C


CANCEL_SHAPED = ["cancel", "cancel that", "cancel it", "cancel the task",
                 "cancel claude", "stop the task", "stop working",
                 "stop claude", "abort", "abort that"]
QUIET_SHAPED = ["stop", "quiet", "be quiet", "never mind", "shush"]


class _Brain:
    def __init__(self):
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1
        return True


class _Claude:
    def __init__(self, running=True):
        self.running, self.cancelled = running, 0

    def cancel(self):
        self.cancelled += 1
        was, self.running = self.running, False
        return was


class _Speech:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1

    interrupt = stop


def _cmdr(brain=None, claude=None, tts=None, reader=None):
    svc = {"brain": brain, "claude": claude, "tts": tts, "reader": reader}
    return SimpleNamespace(_svc=svc.get)


@pytest.mark.parametrize("said", CANCEL_SHAPED + QUIET_SHAPED)
def test_every_stop_word_cancels_the_model_job(said):
    """The whole point: after this, the busy guard is free and his next
    sentence is heard instead of refused."""
    brain, claude, tts = _Brain(), _Claude(), _Speech()
    C._h_quiet(_cmdr(brain, claude, tts), said, None)
    assert brain.cancelled == 1, said


@pytest.mark.parametrize("said", CANCEL_SHAPED + QUIET_SHAPED)
def test_the_speech_in_flight_is_still_cut(said):
    """The behaviour that already worked must keep working."""
    brain, claude, tts = _Brain(), _Claude(), _Speech()
    reader = _Speech()
    C._h_quiet(_cmdr(brain, claude, tts, reader), said, None)
    assert tts.stopped == 1 and reader.stopped == 1, said


@pytest.mark.parametrize("said", CANCEL_SHAPED)
def test_a_cancel_shaped_word_still_stops_claude_and_says_so(said):
    brain, claude, tts = _Brain(), _Claude(running=True), _Speech()
    res = C._h_quiet(_cmdr(brain, claude, tts), said, None)
    assert claude.cancelled == 1, said
    assert res.reply == C.STOPPED_LINE and res.speak is True, said


@pytest.mark.parametrize("said", QUIET_SHAPED)
def test_a_quiet_word_does_not_pretend_it_stopped_a_task(said):
    """"Quiet" is not "cancel the task": it cuts the speech and the model
    job and answers on the panel, never out loud -- he has just asked for
    silence."""
    brain, claude, tts = _Brain(), _Claude(running=True), _Speech()
    res = C._h_quiet(_cmdr(brain, claude, tts), said, None)
    assert claude.cancelled == 0, said
    assert res.speak is False, said


def test_a_brain_that_cannot_cancel_does_not_break_the_stop():
    """A stop must never raise: a missing or broken brain still stops the
    speech and answers."""
    for brain in (None, SimpleNamespace(), SimpleNamespace(cancel="not callable")):
        tts = _Speech()
        res = C._h_quiet(_cmdr(brain, _Claude(), tts), "stop", None)
        assert tts.stopped == 1 and res.handled is True


def test_a_brain_whose_cancel_raises_still_stops_everything_else():
    class _Angry:
        def cancel(self):
            raise RuntimeError("no")
    tts, claude = _Speech(), _Claude(running=True)
    res = C._h_quiet(_cmdr(_Angry(), claude, tts), "cancel that", None)
    assert tts.stopped == 1 and claude.cancelled == 1
    assert res.reply == C.STOPPED_LINE


def test_brain_cancel_is_reachable_from_the_stop_rung_at_all():
    """The census-style pin. Before 09-11 ``brain.cancel`` had exactly one
    caller in this module -- ``_try_correction`` -- so no stop word could
    reach it. If someone removes the call again, this fails."""
    import inspect
    src = inspect.getsource(C._h_quiet)
    assert "_cancel_brain" in src
    assert "cancel" in inspect.getsource(C._cancel_brain)
