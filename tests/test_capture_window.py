"""The follow-up window while a question is open (2026-08-30).

CONFIG.followup_window is 4 s — sized for "...and Tuesday?". It was also
the window for a flashcard answer, so mid-quiz almost every answer needed
the wake word again even though QuizSession.ANSWER_WINDOW_S is 300 s. The
same 4 s covered a yes/no Jarvis had just asked (a destructive read-back,
the good-night wake-alarm offer).

_capture_window() now returns quiz.window_s (15 s) while any of those is
open, on top of the lecture.window_s branch that already existed. The
recorder clamps whatever it is handed to MAX_FOLLOWUP_WINDOW_S.

No app boot: _capture_window only reads self.commander, self.assistant and
self.services, so the app is assembled by hand (the same shape as
tests/test_lecture_notes.py's).
"""
import time
from types import SimpleNamespace

import pytest

import jarvis.app as app_mod
import jarvis.recorder as recorder_mod
from jarvis.commander import DESTRUCTIVE_TTL_S
from jarvis.config import CONFIG
from jarvis.tools.briefing import OFFER_TTL_S
from jarvis.tools.quiz import QuizSession


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(CONFIG, "followup_window", 4.0)
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(
        get=lambda k, d=None: {"lecture.window_s": 20, "quiz.window_s": 15}.get(k, d))
    a.commander = SimpleNamespace(lecture_course=None, _pending_quiz=None,
                                  _pending_destructive=None)
    a.services = SimpleNamespace(alarm_offer=None)
    return a


def _quiz(asked_at=None):
    s = QuizSession([{"id": 1, "question": "q1", "answer": "a1"},
                     {"id": 2, "question": "q2", "answer": "a2"}])
    s.ask(now=asked_at)
    return s


# --------------------------------------------------------------- quiz
def test_an_open_quiz_question_gets_the_longer_window(app):
    assert app._capture_window() is None
    app.commander._pending_quiz = _quiz()
    assert app._capture_window() == 15.0


def test_a_finished_or_stale_quiz_does_not_hold_the_mic_open(app):
    session = _quiz()
    session.index = session.total                       # finished
    app.commander._pending_quiz = session
    assert app._capture_window() is None
    # asked five minutes ago: the session itself would drop it as a new
    # subject, so the window must not stay long either
    app.commander._pending_quiz = _quiz(asked_at=time.time() - 400)
    assert app._capture_window() is None


def test_lecture_notes_still_win_over_a_quiz(app):
    app.commander._pending_quiz = _quiz()
    app.commander.lecture_course = "biosensors"
    assert app._capture_window() == 20.0


# ------------------------------------------------------------- yes/no
def test_a_destructive_read_back_gets_the_longer_window(app):
    app.commander._pending_destructive = (lambda: None, "Cancel all three alarms, sir?",
                                          time.monotonic())
    assert app._capture_window() == 15.0


def test_an_expired_read_back_does_not(app):
    app.commander._pending_destructive = (lambda: None, "line",
                                          time.monotonic() - DESTRUCTIVE_TTL_S - 1)
    assert app._capture_window() is None
    app.commander._pending_destructive = ("junk",)      # never crash on shape
    assert app._capture_window() is None


def test_the_wake_alarm_offer_gets_the_longer_window(app):
    """The offer lives on the services namespace, not on the commander:
    briefing.make_tools parks it there for _try_alarm_offer."""
    app.services.alarm_offer = {"due": 1.0, "time": "7:00 am",
                                "made_at": time.time()}
    assert app._capture_window() == 15.0
    app.services.alarm_offer["made_at"] = time.time() - OFFER_TTL_S - 1
    assert app._capture_window() is None
    app.services.alarm_offer = {}
    assert app._capture_window() is None


# ------------------------------------------------------------ settings
def test_the_window_never_drops_below_the_configured_follow_up(app):
    app.assistant = SimpleNamespace(get=lambda k, d=None: 2)   # below the default
    app.commander._pending_quiz = _quiz()
    assert app._capture_window() == 4.0


def test_a_junk_setting_falls_back_to_the_default(app):
    app.assistant = SimpleNamespace(get=lambda k, d=None: "soon")
    app.commander._pending_quiz = _quiz()
    assert app._capture_window() == 15.0
    app.assistant = None                                # no config at all
    assert app._capture_window() == 15.0


def test_the_recorder_clamps_the_longer_window():
    """The window is a request, not a promise: 15 s is well inside the cap
    the recorder already applies to the lecture window."""
    assert recorder_mod.Recorder.clamp_window(15) == 15.0
    assert 15.0 <= recorder_mod.MAX_FOLLOWUP_WINDOW_S


def test_a_commanderless_app_asks_for_nothing(app):
    app.commander = None
    assert app._capture_window() is None
