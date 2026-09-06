"""A waiting YES/NO card keeps the console awake.

THE HALF OF "STUCK IN STANDBY" THE CAUSE FIX DID NOT TOUCH. On 2026-09-06
he described Jarvis as "stuck from saying is that for me and being in
standby mode". The turn-holding half is fixed (be0f151): an unanswered
"Was that for me?" now releases its turn in ~5 s. But the console dropped
to the clock WHILE the mic was open for his answer and stayed there over
the card, because ``_console_busy`` is ``recording or speaking or
thinking``: ``record_fixed`` publishes no RecordingStarted, so the 5 s
answer window is not "recording" to the window, and a card waiting on
his click is not counted at all. So a question Jarvis is literally
waiting on him to answer put the console to sleep. Named by the
reproduction lane's adversary as its first remaining item; one line.

Tk-free: ``_console_busy`` is bound to a stand-in that owns only the four
attributes it reads, the way tests/test_audio_busy_never_wedges.py binds
JarvisApp methods.
"""
import types

from jarvis.ui import main_window as mw


def _win(**kw):
    w = types.SimpleNamespace(_recording=False, _speaking=False,
                              _thinking=False, _pending=set())
    for k, v in kw.items():
        setattr(w, k, v)
    return w


def _busy(w) -> bool:
    return mw.MainWindow._console_busy.__get__(w)()


def test_a_waiting_card_keeps_the_console_awake():
    """The premise of the incident: nothing else is live, a card waits."""
    assert _busy(_win(_pending={"rid-1"})) is True, (
        "a question Jarvis is waiting on him to answer let the console "
        "fall to standby")


def test_an_idle_console_still_sleeps():
    """The guard must be a guard, not an off switch for standby."""
    assert _busy(_win()) is False


def test_the_three_existing_inputs_are_unchanged():
    assert _busy(_win(_recording=True)) is True
    assert _busy(_win(_speaking=True)) is True
    assert _busy(_win(_thinking=True)) is True


def test_a_card_that_was_resolved_no_longer_holds_it():
    w = _win(_pending={"rid-1"})
    assert _busy(w) is True
    w._pending.discard("rid-1")
    assert _busy(w) is False
