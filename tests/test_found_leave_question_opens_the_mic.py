"""2026-09-02 08:56:15 -- "How long do you need to get to Wisenbaker, sir?"

Two independent defects made one incident, and he reported it as one
sentence: "was stuck at speaking and wouldnt let me respond". He answered
by TYPING 21 s later (`handle 'about 10 minutes' source=typed`).

  1. THE QUESTION OPENED NO MIC. `_ask_leave_time` spoke and returned.
     `_after_speech` starts a follow-up listen only on
     `_followup_after_speech`, which nothing set; `Commander.ask_leave_time`
     arms only `_pending_leave`, the rung that ANSWERS. Every other question
     Jarvis asks arms the listen. This one did not, so there was no
     "Recording started" anywhere in the 08:55-08:57 window of the live log.

  2. THE SPEAKING STATE STUCK ON. See
     tests/test_found_speaking_falling_edge_is_final.py.

Both are pinned here and there rather than in one place because either
alone reproduces "wouldn't let me respond".
"""
import threading
import time
import types

import pytest

from jarvis.app import JarvisApp
from jarvis.commander import LEAVE_ANSWER_WINDOW_S
from jarvis.config import CONFIG


class _Commander:
    """The real arming contract of Commander.ask_leave_time."""

    def __init__(self, refuse=False):
        self._pending_leave = None
        self._pending_session = None
        self.lecture_course = None
        self._refuse = refuse

    def question_open(self):
        return False

    def ask_leave_time(self, key, place):
        if self._refuse:
            return False
        self._pending_leave = (key, place, time.monotonic())
        return True


def _stub(commander=None):
    app = JarvisApp.__new__(JarvisApp)
    app._turn_busy = threading.Event()
    app._audio_busy = threading.Event()
    app.recorder = types.SimpleNamespace(recording=False)
    app.quiet = None
    app._followup_after_speech = False
    app._pending_debrief = None
    app.assistant = {}
    app.said = []
    app._say = lambda text, **kw: app.said.append(text)
    app.commander = commander if commander is not None else _Commander()
    return app


@pytest.fixture
def talkback():
    was = CONFIG.talkback
    CONFIG.talkback = True
    yield
    CONFIG.talkback = was


def test_the_leave_question_arms_the_follow_up_listen(talkback):
    """The incident: it spoke and armed nothing, so typing was the only way."""
    app = _stub()
    assert app._ask_leave_time("wisenbaker", "Wisenbaker") is True
    assert app.said == ["How long do you need to get to Wisenbaker, sir?"]
    assert app._followup_after_speech is True, \
        "the question opened no mic; he had to type the answer"


def test_a_refused_ask_arms_nothing(talkback):
    """No question was put, so no mic: the watch retries on a later tick."""
    app = _stub(_Commander(refuse=True))
    assert app._ask_leave_time("wisenbaker", "Wisenbaker") is False
    assert app.said == []
    assert app._followup_after_speech is False


def test_a_pending_walk_gets_the_question_window_not_the_4s_default(talkback):
    """A walk duration is said after a pause to think, not inside 4 s."""
    app = _stub()
    app._ask_leave_time("wisenbaker", "Wisenbaker")
    assert app._capture_window() == 15.0


def test_the_window_lapses_with_the_answering_rung():
    """The longer window only ever covers a LIVE question."""
    app = _stub()
    app.commander._pending_leave = (
        "wisenbaker", "Wisenbaker",
        time.monotonic() - LEAVE_ANSWER_WINDOW_S - 1.0)
    assert app._capture_window() is None


def test_a_malformed_pending_leave_does_not_break_the_window():
    app = _stub()
    for junk in (None, (), ("wisenbaker",), ("a", "b", "not-a-time")):
        app.commander._pending_leave = junk
        assert app._capture_window() is None


def test_after_speech_actually_starts_the_listen(talkback):
    """The flag is only worth setting if _after_speech consumes it."""
    app = _stub()
    app._briefing_pending = False
    app._pending_uncertain = None
    app.tts = types.SimpleNamespace(is_speaking=False, pending=0)
    started = []
    app._start_followup = lambda: started.append(True)
    app._ask_leave_time("wisenbaker", "Wisenbaker")
    app._after_speech()
    assert started == [True]
    assert app._followup_after_speech is False   # one capture, not hands-free
