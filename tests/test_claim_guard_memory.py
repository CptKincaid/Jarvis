"""The claim guard on a MEMORY-shaped claim: "I have noted that, sir",
"I'll remember that", "I've saved that to memory".

2026-09-02 23:25 he typed "Put it in your memory that I graduate December
10th 2026 with an electrical engineering degree"; the model said "I have
noted that, sir" and stored nothing. Commit 7539478 taught the claim table
those three shapes; the 2026-09-04 refuter then measured, on this same
FakeOllama harness, that the system around the table still let the lie
through:

  * "Do not forget that ..." begins with "do", router.is_question said
    True, the guard never armed, and he heard "I have noted that, sir."
    with zero warnings -- the incident verbatim (probe e);
  * ANY tool in the turn exempted the claim, so "tell me the time and put
    this in your memory" ran get_time and the lie walked through (probe h);
  * the model has NO memory tool, so a guarded turn ended in "I couldn't do
    that part, sir." with the model's honest retry thrown away (probe b),
    or in "Of course, sir." followed by that refusal on the stream (probe f).

Every scenario here is the real JarvisBrain over the FakeOllama seam
(tests/test_brain_tools.py). No Ollama, no network, no audio, no memory
store: the registry is the fake one and nothing it holds stores a fact,
which is also true of the live registry today.
"""
import pytest

import jarvis.brain as brain_mod
from jarvis.router import is_question
from tests.test_brain_tools import (FakeContext, FakeMemory, FakeOllama,  # noqa: F401
                                    brain, make_registry, text_reply, tool_reply)

GRADUATE = ("Do not forget that I graduate December 10th 2026 with an "
            "electrical engineering degree")
NOTED = "I have noted that, sir."


@pytest.fixture
def setup(brain, monkeypatch):  # noqa: F811
    brain.reset_static_prompt()
    record = []
    reg = make_registry(record)
    monkeypatch.setattr(brain, "_REGISTRY", reg)
    fake = FakeOllama()
    monkeypatch.setattr(brain, "_http", fake)
    b = brain.JarvisBrain(context=FakeContext(), memory=FakeMemory())
    return b, fake, record


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelname == "WARNING" and "unbacked" in r.getMessage()]


# ------------------------------------------------- (1) arming: "do not" orders
# router.is_question opened with `do` as a question word, so every
# imperative that begins "Do not ..." / "Do remember ..." disarmed the one
# guard built for it. An utterance that starts with an imperative "do" is
# an ORDER for the guard's purpose; the routing cue (analyse().question)
# is untouched, and the genuine "do you ...?" shapes stay questions.
@pytest.mark.parametrize("text", [
    GRADUATE,
    "Do remember that Heather prefers email",
    "Do keep in mind that the lab moved to room 049",
    "Do note that my locker code is 4412",
    "Do make a note that I graduate December 10th",
    "please do not forget that I graduate December 10th",
    "Jarvis, do not forget that I graduate December 10th",
    # a pin, not a fix: "don't" never matched `do\b`, so it was already an
    # order -- kept here so the family is tested as one
    "Don't forget that I graduate December 10th 2026",
])
def test_an_imperative_do_is_an_order_not_a_question(text):
    assert is_question(text) is False, text


@pytest.mark.parametrize("text", [
    "Do you know what time it is?",
    "Do I have anything on tomorrow?",
    "Does the lab have a listed duration?",
    "Don't you think it's late?",
    "did you set my timer for ten minutes",
    "Do you remember what I said about the thesis?",
    "do we have milk",
])
def test_a_real_do_question_is_still_a_question(text):
    assert is_question(text) is True, text


def test_do_not_forget_is_guarded_on_the_brain(setup, caplog):
    """The 09-02 incident, replayed through the sentence that still
    reached the model: the guard must arm, and "I have noted that, sir."
    must not be what he hears."""
    b, fake, record = setup
    fake.replies = [text_reply(NOTED), text_reply(NOTED)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(GRADUATE)
    spoken = dict(tags)["SPEAK"]
    assert "noted that" not in spoken.lower(), spoken
    assert record == []
    assert _warnings(caplog), "the guard never armed"
    assert len(fake.chat_payloads()) == 2          # the one retry was spent


# ------------------------------------------------ (5) the memory shapes
# 7539478 taught the table "I have noted that", "I'll remember/note that"
# and "I've saved that to memory". The refuter listed the promises that
# still walked through (store, keep/bear in mind, make a note, the
# have-less "I noted that already", "I've remembered that") and the
# ordinary English that must not be caught.
@pytest.mark.parametrize("line", [
    "I'll keep that in mind, sir.",
    "I'll bear that in mind, sir.",
    "I'll make a note of that.",
    "I'll store that for you.",
    "I've remembered that, sir.",
    "I noted that already, sir.",
    "I will keep this in mind.",
    "I shall make a note of it, sir.",
    "I've made a note of that, sir.",
    "I'm making a note of that now, sir.",
    "I won't forget that, sir.",              # the negative promise IS the promise
    "I've committed that to memory, sir.",
    "I have noted that, sir; December 10th for your graduation.",
])
def test_a_memory_promise_is_an_unbacked_claim(line):
    assert brain_mod.unbacked_claim(line), line


@pytest.mark.parametrize("line", [
    "I've saved you twenty minutes, sir.",          # saved + a duration: an idiom
    "I've saved you some time, sir.",
    "I've saved you the trouble of a second trip.",
    "I've saved you a trip to the lab, sir.",
    "As I noted earlier, the lab has no listed duration.",
    "As I noted this morning, your flight is at nine.",
    "You noted that yourself last week, sir.",
    "It is noted in your calendar as a tentative hold.",
    "I remember that day well, sir.",               # present tense, no promise
    "I can't remember that, sir; it was before my time.",
    "I haven't noted anything about your graduation, sir.",
    "Shall I remember that for you, sir?",          # an offer
    "Would you like me to keep that in mind?",
    "I'll keep it short, sir.",                     # keep, but not in mind
    "I'll remember this evening for a long time, sir.",
    "I'm afraid I have no way to store that, sir; say remember that and I shall.",
])
def test_ordinary_memory_talk_is_not_a_claim(line):
    assert brain_mod.unbacked_claim(line) is None, line


def test_the_authored_memory_line_is_not_itself_a_claim():
    """The line the guard speaks for a memory claim carries the words
    'remember that' and 'I will'; a retry that echoes it must not trip
    the table it was written to answer."""
    assert brain_mod.unbacked_claim(brain_mod.UNBACKED_MEMORY_LINE) is None
