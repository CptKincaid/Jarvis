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


# ------------------------------------------- (2) what backs a memory claim
# The guard's exemption was "any tool ran this turn". A memory claim is
# backed by a tool that STORES A FACT, and no registry tool does (the
# remember rung lives in the commander, which the model cannot call), so
# "tell me the time and put this in your memory" ran get_time and "I have
# noted that you graduate December 10th" walked through (probe h). The
# table below is the one place a future memory tool is named.
COMPOUND = ("Tell me the time and put this in your memory: I graduate "
            "December 10th 2026")
TIME_AND_LIE = "It's five past four, sir. I have noted that you graduate December 10th."


def test_no_registry_tool_backs_a_memory_claim_today():
    assert brain_mod.CLAIM_BACKERS["memory"] == frozenset()
    assert brain_mod.CLAIM_BACKERS["action"] is None        # any tool at all
    assert brain_mod.claim_kind("I have noted that") == "memory"
    assert brain_mod.claim_kind("I'll keep that in mind") == "memory"
    assert brain_mod.claim_kind("I've added") == "action"
    assert brain_mod.claim_backed("I have noted that", ("get_time", "notes")) is False
    assert brain_mod.claim_backed("I've added", ("notes",)) is True
    assert brain_mod.claim_backed("I've added", ()) is False


def test_a_tool_that_ran_backs_an_action_claim_but_not_a_memory_one():
    assert brain_mod.unbacked_claim("I've added milk to your list.", ran=("notes",)) is None
    assert brain_mod.unbacked_claim(TIME_AND_LIE, ran=("get_time",)) == "I have noted that"
    # the one-place extension: a fact-storing tool named in the table backs it
    assert brain_mod.unbacked_claim(TIME_AND_LIE, ran=("get_time",),
                                    backers={"memory": frozenset({"get_time"})}) is None


def test_a_memory_claim_beside_a_real_tool_is_still_caught(setup, caplog):
    """get_time runs, the model reads the time AND claims the store: the
    time is his, the store is a lie, and the lie must not be spoken."""
    b, fake, record = setup
    fake.replies = [tool_reply(("get_time", {})),
                    text_reply(TIME_AND_LIE),
                    text_reply("It's five past four, sir.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(COMPOUND)
    spoken = dict(tags)["SPEAK"]
    assert "noted" not in spoken.lower(), spoken
    assert "five past four" in spoken
    assert record == [("get_time", None)]                     # once, not again
    assert _warnings(caplog)[0] == (
        "brain: unbacked memory claim 'I have noted that' (get_time ran; "
        "nothing that ran stores a fact)")
    assert len(fake.chat_payloads()) == 3                      # the one retry


# ------------------------------- a question that also asks for a store
# "Tell me the time and put this in your memory: ..." is a QUESTION by the
# router's rule (tell me), and the guard stands down on a question because
# its retry executes. A memory claim cannot execute anything real (no tool
# stores a fact), so on a question that also asks for a store the guard
# judges MEMORY claims only -- an action claim on the same question stays
# the answer it is, the way the 09-02 timer bug demands.
def test_a_question_that_asks_for_a_store_judges_memory_claims_only(setup, caplog):
    b, fake, record = setup
    asked = "What's the weather, and remember that I graduate December 10th"
    assert is_question(asked)
    fake.replies = [tool_reply(("get_weather", {"when": "now"})),
                    text_reply("Seventy-two and cloudy, sir. I'll remember that."),
                    text_reply("Seventy-two and cloudy, sir.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(asked)
    assert dict(tags)["SPEAK"] == "Seventy-two and cloudy, sir."
    assert record == [("get_weather", "now", None)]
    assert len(_warnings(caplog)) == 1


def test_an_action_claim_on_a_memory_question_is_still_never_acted_on(setup, caplog):
    """"Do you remember if you added milk?" asks for nothing to be stored
    and answers with an action claim: the retry would WRITE. Untouched."""
    b, fake, record = setup
    fake.replies = [text_reply("Yes, sir. I've added milk to your list already."),
                    tool_reply(("notes", {"action": "add", "text": "milk"}))]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("Do you remember if you added milk to my list?")
    assert tags == [("SPEAK", "Yes, sir. I've added milk to your list already.")]
    assert record == [] and len(fake.chat_payloads()) == 1
    assert _warnings(caplog) == []


# ------------------------------------------ (3) the honest retry is kept
# brain.py discarded the retry by design ("the FIRST reply is what he
# hears, with the claims taken out"), so the model's one honest answer --
# "I'm afraid I have no way to store that, sir" -- was thrown away for
# "I couldn't do that part, sir." (probe b). The reason no longer holds:
# the retry is judged by the same table, so a retry that claims nothing
# and ran no tool is the model answering the nudge as written, and it is
# the only path to an honest sentence in the model's own words.
HONEST = "I'm afraid I have no way to store that, sir; say remember that and I shall."
STORE = "Put this in your memory: I graduate December 10th 2026"


def test_a_claim_free_retry_is_spoken_not_discarded(setup, caplog):
    b, fake, record = setup
    fake.replies = [text_reply(NOTED), text_reply(HONEST)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(STORE)
    assert tags == [("SPEAK", HONEST)]
    assert record == []
    assert _warnings(caplog) == [
        "brain: unbacked action claim 'I have noted that' (no tool ran)"]
    assert len(fake.chat_payloads()) == 2


def _chunks(text):
    out = [{"message": {"role": "assistant", "content": text[i:i + 7]}, "done": False}
           for i in range(0, len(text), 7)]
    out.append({"message": {"role": "assistant", "content": ""}, "done": True,
                "load_duration": 0})
    return out


@pytest.fixture
def streamed(setup, monkeypatch):
    b, fake, record = setup
    streams = []

    def stream(path, payload, timeout=None):
        assert streams, "no scripted stream left"
        for chunk in streams.pop(0):
            yield chunk
    monkeypatch.setattr(brain_mod, "_http_stream", stream)
    return b, fake, record, streams


GREETING = "Good evening, Ali and Heather."


def test_a_claim_free_retry_is_spoken_after_the_streamed_greeting(streamed, caplog):
    """The greeting went to TTS as it landed and the claim was withheld;
    the honest retry follows it, once, on the same stream."""
    b, fake, record, streams = streamed
    streams.append(_chunks(f"{GREETING} I'll remember that, sir."))
    fake.replies = [text_reply(HONEST)]
    spoken = []
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("say hello to my family and remember that I graduate "
                            "December 10th", on_sentence=spoken.append)
    assert spoken == [GREETING, HONEST]
    assert tags == [("STREAMED", "2"), ("SPEAK", f"{GREETING} {HONEST}")]
    assert record == [] and len(_warnings(caplog)) == 1


# --------------------------------- (4) the line he hears, and nothing before it
# For a memory claim the reply is UNBACKED_MEMORY_LINE -- it names the way
# in -- and a bare acknowledgement ("Of course, sir.") that was the yes to
# the claim goes with the claim: a yes followed by a refusal, spoken as one
# reply, is what the stream produced (probe f). What he asked for beside
# the claim (a greeting, a tool's answer) is kept: it is not a lead-in.
MEMORY_LINE = brain_mod.UNBACKED_MEMORY_LINE
OF_COURSE = "Of course, sir. I'll remember that."


def test_the_memory_line_is_the_whole_reply(setup):
    b, fake, record = setup
    fake.replies = [text_reply(NOTED), text_reply(NOTED)]
    assert b._chat_sync(STORE) == [("SPEAK", MEMORY_LINE)]
    assert record == []


def test_no_of_course_lead_in_before_the_refusal(setup):
    b, fake, record = setup
    fake.replies = [text_reply(OF_COURSE), text_reply(OF_COURSE)]
    assert b._chat_sync(STORE) == [("SPEAK", MEMORY_LINE)]


def test_a_greeting_and_a_tools_answer_are_kept_beside_the_line(setup):
    b, fake, record = setup
    fake.replies = [text_reply(f"{GREETING} I'll remember that, sir."),
                    text_reply(f"{GREETING} I'll remember that, sir.")]
    assert b._chat_sync("say hello to my family and remember that I graduate "
                        "December 10th") == [("SPEAK", f"{GREETING} {MEMORY_LINE}")]
    fake.replies = [tool_reply(("get_time", {})), text_reply(TIME_AND_LIE),
                    text_reply(TIME_AND_LIE)]
    assert b._chat_sync(COMPOUND) == [("SPEAK", f"It's five past four, sir. {MEMORY_LINE}")]


def test_each_kind_gets_its_own_line_once_in_place():
    strip = brain_mod.strip_unbacked_claims
    assert strip(OF_COURSE) == MEMORY_LINE
    assert strip(f"{GREETING} I'll remember that. Enjoy your evening.") == \
        f"{GREETING} {MEMORY_LINE} Enjoy your evening."
    assert strip("I've added milk. I'll remember that. I've noted it too.") == \
        f"{brain_mod.UNBACKED_LINE} {MEMORY_LINE}"
    assert strip("Certainly, sir. I've added milk to your list.") == brain_mod.UNBACKED_LINE
    # the cap keeps every authored line, at the cost of trailing prose
    assert strip("One. I've added milk. I'll remember that. Four.", 2) == \
        f"{brain_mod.UNBACKED_LINE} {MEMORY_LINE}"
    assert strip("One. Two. I'll remember that. Four.", 2) == f"One. {MEMORY_LINE}"


def test_a_memory_claim_gets_the_memory_nudge(setup):
    """"Use the tools and do it now" steered a store the model cannot make
    toward the notes tool (a write the recall path never reads). The
    memory nudge says there is no such tool and not to invent one; whether
    gemma obeys is his to measure, no Ollama here."""
    b, fake, record = setup
    fake.replies = [text_reply(NOTED), text_reply(NOTED)]
    b._chat_sync(STORE)
    p1, p2 = fake.chat_payloads()
    assert p2["messages"][-1]["content"] == brain_mod.UNBACKED_MEMORY_NUDGE
    assert p2["messages"][-2]["content"] == NOTED
    assert "notes" in brain_mod.UNBACKED_MEMORY_NUDGE.lower()
    assert p2["messages"][0] == p1["messages"][0]        # the static prefix untouched


def test_streamed_of_course_does_not_stand_before_the_line(streamed, caplog):
    """The lead-in lands on the stream BEFORE the claim it is the yes to:
    it is held one sentence, dropped when the claim follows, and he hears
    the authored line alone."""
    b, fake, record, streams = streamed
    streams.append(_chunks(OF_COURSE))
    fake.replies = [text_reply(OF_COURSE)]
    spoken = []
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(STORE, on_sentence=spoken.append)
    assert spoken == [MEMORY_LINE]
    assert tags == [("STREAMED", "1"), ("SPEAK", MEMORY_LINE)]
    assert len(_warnings(caplog)) == 2


def test_a_held_lead_in_is_released_with_honest_content(streamed):
    b, fake, record, streams = streamed
    streams.append(_chunks("Of course, sir. Milk and eggs, sir."))
    spoken = []
    tags = b._chat_sync("read me my shopping list", on_sentence=spoken.append)
    assert spoken == ["Of course, sir.", "Milk and eggs, sir."]
    assert tags == [("STREAMED", "2"), ("SPEAK", "Of course, sir. Milk and eggs, sir.")]
    assert fake.chat_payloads() == []                     # no retry


def test_a_lead_in_alone_is_spoken_when_the_round_ends(streamed):
    b, fake, record, streams = streamed
    streams.append(_chunks("Of course, sir."))
    spoken = []
    tags = b._chat_sync("read me my shopping list", on_sentence=spoken.append)
    assert spoken == ["Of course, sir."]
    assert tags == [("STREAMED", "1"), ("SPEAK", "Of course, sir.")]


def test_a_lead_in_before_a_tool_call_stays_unspoken_and_the_tool_runs(streamed):
    """A pin, not a change: a round that turns into a tool call speaks
    nothing (_stream_round), so "Of course, sir." before the call was
    never spoken and the hold must not start speaking it -- the tool's
    own line is the reply."""
    b, fake, record, streams = streamed
    chunks = _chunks("Of course, sir. ")[:-1]
    chunks.append({"message": {"role": "assistant", "content": "",
                               "tool_calls": [{"function": {
                                   "name": "notes",
                                   "arguments": {"action": "add", "text": "milk"}}}]},
                   "done": True, "load_duration": 0})
    streams.append(chunks)
    spoken = []
    tags = b._chat_sync("add milk to my list", on_sentence=spoken.append)
    assert spoken == []
    assert record == [("notes", "add", "milk")]
    assert tags == [("SPEAK", "Noted, sir.")]
