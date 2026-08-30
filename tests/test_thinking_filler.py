"""The slow-lookup filler must not answer a question Jarvis just asked.

"Was that for me?" leaves the turn open (done=False) so a yes/no can arrive,
which left the THINKING_DELAY_S timer running. Live at 21:08 on 2026-08-29:

    08.865 Uncertain intent -> "Was that for me?"
    12.365 speaking "Looking into it now, sir."     <- 3.5 s later, unanswered
    12.570 status: Discarded

The user heard their question answered with a promise to work on an utterance
that was then thrown away. By 12.365 _ask_uncertain is also recording the
spoken reply, and the mic arbiter is a depth counter rather than a mutex, so
Jarvis was talking into his own yes/no window.
"""
import threading

from jarvis.app import JarvisApp


def _app(pending):
    a = object.__new__(JarvisApp)
    a._turn_busy = threading.Event()
    a._turn_busy.set()
    a._uncertain_lock = threading.Lock()
    a._pending_uncertain = dict(pending)
    a._thinking_i = 0
    a.said = []
    a._say = a.said.append
    return a


def test_held_while_a_question_is_open():
    app = _app({"abc123": "what have you, and"})
    app._say_thinking()
    assert app.said == [], f"spoke while awaiting a yes/no: {app.said}"


def test_still_speaks_for_a_genuinely_slow_lookup():
    """The filler must keep working; this is a suppression, not a removal."""
    app = _app({})
    app._say_thinking()
    assert len(app.said) == 1 and app.said[0]


def test_silent_once_the_answer_has_landed():
    app = _app({})
    app._turn_busy.clear()
    app._say_thinking()
    assert app.said == []


def test_resumes_after_the_question_is_answered():
    """Answering must not leave the filler permanently muted."""
    app = _app({"abc123": "what have you, and"})
    app._say_thinking()
    assert app.said == []
    app._pending_uncertain.clear()          # user clicked yes/no
    app._say_thinking()
    assert len(app.said) == 1


def test_the_delay_clears_the_measured_tool_latency():
    """Tool-backed lookups landed right at 3.5 s, so the filler started and
    the real answer queued behind it."""
    import jarvis.app as app_mod
    assert app_mod.THINKING_DELAY_S >= 4.5
