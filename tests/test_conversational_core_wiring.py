"""The app half of the Aside and the debrief (jarvis/app.py).

Both features live outside app.py; what is wired here is the small set of
decisions the app owns and nothing else can test:

  * the aside is spoken as a SEPARATE utterance AFTER the reply, never
    spliced into it, and only when the handler actually produced a
    structured action result to anchor on;
  * an open debrief question owns the next transcript -- but gives it up
    for anything that is plainly a command, and for a wake word;
  * a debrief answer is filed, never routed to commander.handle.

The app is built with object.__new__ and only the seams each path
touches, exactly as tests/test_destructive_readback.py does. No Tk, no
audio, no threads.
"""
import threading
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import jarvis.app as app_mod
from jarvis.commander import CommandResult
from jarvis.debrief import Candidate

TZ = ZoneInfo("America/Chicago")
DAY = datetime(2026, 9, 14, tzinfo=TZ)


@pytest.fixture(autouse=True)
def _talkback(monkeypatch):
    # _ask_debrief is a speech path and refuses outright without talk-back;
    # pinned rather than assumed so a CONFIG default change fails loudly
    # here instead of silently skipping every debrief assertion.
    monkeypatch.setattr(app_mod.CONFIG, "talkback", True)


def _app(handle=None):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: d, user_name="Hunter")
    a._init_assistant_state()
    a.said = []
    a._say = a.said.append
    a.exchanges = []
    a.context = SimpleNamespace(add_exchange=lambda u, j: a.exchanges.append((u, j)))
    a.commander = SimpleNamespace(
        handle=handle or (lambda text, source, **kw: CommandResult(handled=True)),
        _match_assistant=lambda t: None)
    a.turns = SimpleNamespace(mark=lambda *x, **k: None, abandon=lambda r: None)
    a._turn_start = lambda: None
    a._turn_after_result = lambda r: None
    a._turn_finished = lambda: None
    a._turn_busy = threading.Event()
    a._audio_busy = threading.Event()
    a.recorder = SimpleNamespace(recording=False)
    a.tts = SimpleNamespace(is_speaking=False, pending=0)
    return a


# --------------------------------------------------------------- the aside
def test_the_aside_follows_the_answer_as_its_own_utterance():
    """Never spliced into the reply: TTS.speak is a FIFO worker, so the
    answer plays first and the aside lands behind it as a separate beat --
    which is what makes it sound like an afterthought."""
    item = SimpleNamespace(kind="alarm", due=0.0)

    def handle(text, source, **kw):
        return CommandResult(handled=True, reply="Alarm for 7:00 am, sir.",
                             speak=True, status="Alarm", action=item)
    a = _app(handle)
    seen = []
    a.aside = SimpleNamespace(consider=lambda t, r, act: seen.append((t, r, act))
                              or "Incidentally, Lab 3 report is due at 11:59 pm.")
    a._dispatch("wake me at seven", "typed")
    assert a.said == ["Alarm for 7:00 am, sir.",
                      "Incidentally, Lab 3 report is due at 11:59 pm."]
    assert seen[0][2] is item          # the STRUCTURED result, not the words


def test_no_action_result_never_reaches_the_engine():
    a = _app(lambda text, source, **kw: CommandResult(
        handled=True, reply="It's 8:41 am, sir.", speak=True))
    a.aside = SimpleNamespace(
        consider=lambda *x: pytest.fail("consider() was called with no anchor"))
    a._dispatch("what time is it", "typed")
    assert a.said == ["It's 8:41 am, sir."]


def test_an_engine_that_raises_never_breaks_the_turn():
    item = SimpleNamespace(kind="alarm", due=0.0)
    a = _app(lambda text, source, **kw: CommandResult(
        handled=True, reply="Alarm set, sir.", speak=True, action=item))
    a.aside = SimpleNamespace(consider=lambda *x: 1 / 0)
    a._dispatch("wake me at seven", "typed")
    assert a.said == ["Alarm set, sir."]


def test_no_engine_at_all_is_simply_quiet():
    item = SimpleNamespace(kind="alarm", due=0.0)
    a = _app(lambda text, source, **kw: CommandResult(
        handled=True, reply="Alarm set, sir.", speak=True, action=item))
    a._dispatch("wake me at seven", "typed")
    assert a.said == ["Alarm set, sir."]


# ------------------------------------------------------------- the debrief
def _cand():
    return Candidate("BIOSENSORS Midterm 1|x", "BIOSENSORS Midterm 1", "midterm",
                     DAY.replace(hour=14, minute=30))


def _armed(handle=None):
    a = _app(handle)
    a.memory = SimpleNamespace(remember=lambda k, v: None)
    filed = []
    a.context = SimpleNamespace(
        journal_debrief=lambda *args: filed.append(args),
        add_exchange=lambda u, j: None)
    a.debrief = SimpleNamespace(mark_asked=lambda key, now=None: None)
    a._ask_debrief(_cand())
    return a, filed


def test_the_question_is_asked_once_and_the_ledger_is_written_at_once():
    """Written when the question is PUT, not when it is answered: an
    unanswered debrief has still been asked."""
    marked = []
    a = _app()
    a.debrief = SimpleNamespace(mark_asked=lambda key, now=None: marked.append(key))
    a._ask_debrief(_cand())
    assert a.said == ["How did the midterm go, sir?"]
    assert marked == ["BIOSENSORS Midterm 1|x"]
    assert a._followup_after_speech is True     # answerable without a wake word


def test_the_next_transcript_is_filed_not_routed():
    def handle(text, source, **kw):
        pytest.fail("a debrief answer was routed to the commander")
    a, filed = _armed(handle)
    a.said.clear()
    res = a._dispatch("it went badly, I ran out of time", "typed")
    assert res.reply == a.DEBRIEF_FILED_LINE
    assert filed[0][:3] == ("BIOSENSORS Midterm 1", "midterm",
                            "it went badly, I ran out of time")
    assert a._pending_debrief is None


def test_a_plain_command_keeps_its_own_meaning():
    """"Set a timer for five minutes" is not how the midterm went, and
    filing it as such would poison a record he is meant to trust."""
    routed = []
    a, filed = _armed(lambda text, source, **kw: routed.append(text) or
                      CommandResult(handled=True, status="Timer set"))
    a.commander._match_assistant = lambda t: "timer"
    a._dispatch("set a timer for five minutes", "typed")
    assert routed == ["set a timer for five minutes"] and filed == []


def test_the_jarvis_prefix_also_gives_the_turn_back():
    routed = []
    a, filed = _armed(lambda text, source, **kw: routed.append(text) or
                      CommandResult(handled=True))
    a._dispatch("jarvis what time is it", "typed")
    assert routed and filed == []


def test_a_decline_files_nothing_and_is_still_never_asked_again():
    a, filed = _armed()
    res = a._dispatch("no, never mind", "typed")
    assert res.reply == a.DEBRIEF_DECLINED_LINE and filed == []


def test_a_late_answer_is_not_an_answer():
    a, filed = _armed(lambda text, source, **kw: CommandResult(handled=True))
    a._pending_debrief["at"] -= a.DEBRIEF_TTL_S + 1
    a._dispatch("it went fine", "typed")
    assert filed == []


def test_one_open_question_at_a_time():
    a = _app()
    a.debrief = SimpleNamespace(mark_asked=lambda key, now=None: None)
    a._ask_debrief(_cand())
    a.said.clear()
    a._ask_debrief(_cand())
    assert a.said == []


def test_he_is_never_asked_over_a_live_turn():
    a = _app()
    a.debrief = SimpleNamespace(mark_asked=lambda key, now=None: None)
    a._turn_busy.set()
    a._ask_debrief(_cand())
    assert a.said == [] and a._pending_debrief is None
