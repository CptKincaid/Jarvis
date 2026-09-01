"""FOUND 2026-08-31 (Hunter's 155-feature evening, /tmp/vss_voice/jarvis.log):
three routing defects, each quoted from his own note and each traced to a
turn in the log.

#9  "quiet" -- "does not understand" (also "hush", "be quiet", "stop
    talking").  A bare barge-in word carries no "jarvis" prefix, so
    ``strip_jarvis_prefix`` returns None, the whole registry rung
    (``Command("quiet", quiet_kind, _h_quiet)``) is SKIPPED, and the
    intent gate below it -- which calls every one- and two-word phrase
    background chat -- dropped it in silence.  Measured on the real
    classifier before the fix:

        classify("quiet")         -> ("no", 0.80)  -> Ignored
        classify("hush")          -> ("no", 0.80)  -> Ignored
        classify("be quiet")      -> ("no", 0.80)  -> Ignored
        classify("shut up")       -> ("no", 0.80)  -> Ignored
        classify("that's enough") -> ("no", 0.80)  -> Ignored
        classify("stop talking")  -> ("yes", 0.90) -> Quiet   (3 words)

    Only the three-word one worked, which is exactly why it looked like
    "does not understand" rather than "never fires".  Nobody prefixes the
    word they are using to interrupt.

#95 web lookups -- "When is Texas A&M playing Arizona? / I'm afraid I
    don't know when those teams are playing, sir." then when he pushed
    back it answered correctly.  Log, 21:24:59:

        route local (local:question) 'When is Texas A&M playing Arizona?'
        chat reply: I'm afraid I don't know when those teams are playing, sir.
        route local (local:question) 'When is Texas A&M playing Arizona State University?'
        chat reply: I'm afraid I still don't have the schedule for those teams, sir.
        route web (web-cue) 'Yeah, so you should be able to look that up'
        web answer (haiku): You're right, sir. Texas A&M plays Arizona State
                            on September twelfth at eleven ...

    The lookup works.  What refused was the ROUTER: _WEB_CUE_RX had a
    release-date branch ("when does X come out") but no fixture branch, so
    a kickoff time -- a current fact no local tool holds -- went to the
    3B model, which made something up twice.  The web only ran because he
    said the magic words "look that up".

#114 follow-ups -- "and the next day" returned the SAME day's events
    verbatim rather than tomorrow's.  Log, 21:30:37 and 21:30:50:

        route local (local:question) 'What do I have going on tomorrow?'
        tool get_calendar -> ok=True Tomorrow: 12:45 pm BIOSENSORS ...
        chat reply: You have BIOSENSORS at 12:45 pm, a chiropractor ...
        Uncertain intent (conf=0.50): 'and the next day'      <- asked him
        route local (short) 'and the next day'
        chat reply: Tomorrow you have BIOSENSORS at 12:45 pm, a chiropractor ...

    No get_calendar call on the second turn: the model replayed its own
    previous answer.  The very next utterance proves the model can do the
    arithmetic when it has a sentence to do it on --

        route local (classify) 'and what about the day after that?'
        tool get_calendar -> ok=True Wednesday: 9:10 am BIOSENSORS ...

    -- so the defect is that a four-word fragment reached the model with
    no day in it at all.
"""
import datetime
import types
from unittest.mock import MagicMock

import pytest

from jarvis.commander import (
    Commander,
    IntentClassifier,
    day_shift_followup,
)
from jarvis.config import CONFIG
from jarvis.router import Router


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    """A commander with the real router and intent classifier -- the two
    things every one of these defects lives in -- and mocks for the rest."""
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", False)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(),
        router=Router(None, classify=None), claude=None)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    return Commander(svc)


# ------------------------------------------------------------------ #9
# Every one of these is how he actually interrupts: no wake word left in
# the transcript (the hotword ate it) and no "jarvis" typed in front.
BARGE_IN = ["quiet", "hush", "be quiet", "shut up", "that's enough",
            "stop talking", "stop", "shush", "pipe down", "enough"]


@pytest.mark.parametrize("text", BARGE_IN)
def test_bare_barge_in_cuts_the_speech_instead_of_being_dropped(cmdr, text):
    """#9: "quiet" -- "does not understand"."""
    res = cmdr.handle(text, source="voice")
    assert res.status == "Quiet", (text, res.status)
    # Shown, never spoken: he just told Jarvis to stop making noise.
    assert res.speak is False
    cmdr.services.tts.interrupt.assert_called()


# The three he named first are the ones the classifier rejects outright;
# pin that, so if this test ever fails it says "the cause moved", not
# "the fix broke".
@pytest.mark.parametrize("text", ["quiet", "hush", "be quiet", "shut up",
                                  "that's enough"])
def test_the_intent_gate_is_what_used_to_eat_them(cmdr, text):
    intent, _conf = cmdr.intent.classify(text)
    assert intent == IntentClassifier.NO, (text, intent)
    # ... and the barge-in rung now runs ahead of that gate.
    assert cmdr.handle(text, source="voice").status == "Quiet"


def test_a_question_about_being_quiet_is_still_a_question(cmdr):
    """He also said "Why are you being quiet?" that evening (20:33:19) and
    got a real answer.  _QUIET_RX is anchored, so the new rung must not
    swallow it."""
    res = cmdr.handle("Why are you being quiet?", source="voice")
    assert res.status != "Quiet"
    cmdr.services.brain.chat.assert_called_once()


# ----------------------------------------------------------------- #95
FIXTURES = [
    "When is Texas A&M playing Arizona?",                  # his words, 21:24:59
    "When is Texas A&M playing Arizona State University?",  # his words, 21:25:34
    "who is Texas A&M playing this weekend",
    "when do the Aggies play next",
]


@pytest.mark.parametrize("text", FIXTURES)
def test_a_fixture_question_goes_to_the_web(text):
    """#95: the first attempt refused.  A kickoff time is a current fact
    no local tool holds; the shape of the question is the cue, and he
    should not have to say "look that up" to get it."""
    d = Router(None, classify=None).route(text)
    assert d.kind == "web", (text, d.kind, d.reason)


# The web branch is "kept narrow on purpose" (router.py).  These wear the
# same words and belong to his own diary, his music or his terminal.
NOT_THE_WEB = [
    "what's playing",
    "what's playing right now",
    "what song is playing",
    "when is my next class",
    "when is my meeting",
    "when is my next exam",
    "what's on my calendar today",
    "what is due this week",
    "when is my alarm",
    "when does my lecture start",
    "who is on the call",
    "play some jazz",
]


@pytest.mark.parametrize("text", NOT_THE_WEB)
def test_his_own_things_stay_local(text):
    d = Router(None, classify=None).route(text)
    assert d.kind != "web", (text, d.kind, d.reason)


# ---------------------------------------------------------------- #114
# Monday 2026-08-31 -- the evening of the session, so the weekday names
# below are the ones he would have heard.
MONDAY = datetime.date(2026, 8, 31)


@pytest.mark.parametrize("prev,follow,want", [
    # His two turns, verbatim.
    ("What do I have going on tomorrow?", "and the next day",
     "What do I have going on Wednesday?"),
    ("What do I have going on tomorrow?", "and what about the day after that?",
     "What do I have going on Wednesday?"),
    # The same shape asked the other ways.
    ("What's on my calendar tomorrow?", "the next day",
     "What's on my calendar Wednesday?"),
    ("What's on my calendar today?", "and the next day",
     "What's on my calendar tomorrow?"),
    ("What's the weather tomorrow?", "and the following day",
     "What's the weather Wednesday?"),
    ("What's on my calendar Friday?", "and the next day",
     "What's on my calendar Saturday?"),
])
def test_a_day_shift_follow_up_re_asks_the_question_one_day_on(prev, follow, want):
    assert day_shift_followup(prev, follow, MONDAY) == want


@pytest.mark.parametrize("prev,follow", [
    # No day in the question before it: nothing to move.
    ("What's the weather?", "and the next day"),
    ("", "and the next day"),
    # Not a bare fragment -- a sentence that happens to start the same way.
    ("What's on my calendar tomorrow?", "the next day I'm free"),
    # Not this shape at all.
    ("What's on my calendar tomorrow?", "what's the weather"),
    ("What's on my calendar tomorrow?", "and the day before"),
    # Seven days out: "Monday" would name TODAY to every downstream day
    # parser, so refuse rather than answer about the wrong day.
    ("What's on my calendar Sunday?", "and the next day"),
])
def test_anything_else_is_left_alone(prev, follow):
    assert day_shift_followup(prev, follow, MONDAY) is None


def test_the_follow_up_reaches_the_model_with_a_day_in_it(cmdr):
    """#114: end to end.  Before the fix the second turn was called
    UNCERTAIN by the gate and then handed to the model as the bare
    fragment "and the next day", which replayed tomorrow's list."""
    cmdr.handle("What do I have going on tomorrow?", source="voice")
    assert cmdr.services.brain.chat.call_args[0][0] == \
        "What do I have going on tomorrow?"
    cmdr.services.brain.chat.reset_mock()

    res = cmdr.handle("and the next day", source="voice")
    assert res.status != "Was that for me?"
    sent = cmdr.services.brain.chat.call_args[0][0]
    assert sent != "and the next day"
    assert "going on" in sent          # his subject is kept
    assert "tomorrow" not in sent.lower()   # ... and the day has moved on
    # The exchange is recorded under the resolved sentence, so a SECOND
    # "and the next day" has a day to move rather than a fragment.
    assert res.corrected == sent
