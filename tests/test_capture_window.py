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


# ===================================== Commander.question_open (2026-08-31)
# _question_open grew its own half-list of pending questions and missed the
# ones the later waves added: the exam-week study offer ("Shall we run ten
# now, sir?") and the objection ("Shall I set it anyway?") are both spoken
# yes/no questions Jarvis asked, and both got the 4 s window instead of 15 s.
# The predicate now lives on the Commander, where the rungs that own the
# floor live, so a caller cannot forget one -- app._question_open consults
# it, and so does Commander.ask_leave_time.
class _Svc(SimpleNamespace):
    pass


def _commander(**svc):
    from jarvis.commander import Commander
    c = object.__new__(Commander)
    c.services = _Svc(**svc)
    return c


def test_a_bare_commander_has_no_question_open():
    """object.__new__ shape: every rung is read with getattr, so a slim
    commander answers False rather than raising -- tests/test_custom_phrases.py
    builds one with no services namespace at all."""
    from jarvis.commander import Commander
    assert _commander().question_open() is False
    assert object.__new__(Commander).question_open() is False


def test_an_open_quiz_is_a_question():
    c = _commander()
    c._pending_quiz = _quiz()
    assert c.question_open() is True
    c._pending_quiz = _quiz(asked_at=time.time() - 400)     # stale
    assert c.question_open() is False


def test_an_open_working_session_is_a_question():
    c = _commander()
    c._pending_session = SimpleNamespace(finished=False, stale=lambda: False)
    assert c.question_open() is True
    c._pending_session = SimpleNamespace(finished=True, stale=lambda: False)
    assert c.question_open() is False


def test_a_read_back_and_an_objection_are_both_questions():
    c = _commander()
    c._pending_destructive = (lambda: None, "Cancel all three, sir?",
                              time.monotonic())
    assert c.question_open() is True
    c._pending_destructive = None
    # "Shall I set it anyway?" -- a 4-tuple with its stamp last
    c._pending_objection = (lambda: None, "Shall I set it anyway, sir?",
                            object(), time.monotonic())
    assert c.question_open() is True
    c._pending_objection = (lambda: None, "line", object(),
                            time.monotonic() - DESTRUCTIVE_TTL_S - 1)
    assert c.question_open() is False
    c._pending_objection = ("junk",)                        # never crash on shape
    assert c.question_open() is False


def test_both_briefing_offers_are_questions():
    """The wake-alarm offer AND the exam-week study offer are parked on the
    services namespace by briefing.make_tools; the study one was the miss."""
    for name in ("alarm_offer", "study_offer"):
        c = _commander(**{name: {"made_at": time.time(), "n": 10}})
        assert c.question_open() is True, name
        getattr(c.services, name)["made_at"] = time.time() - OFFER_TTL_S - 1
        assert c.question_open() is False, name
        setattr(c.services, name, {})
        assert c.question_open() is False, name
