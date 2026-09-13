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
from jarvis import arc as arc_mod
from jarvis.router import is_question
from tests.test_brain_tools import (FakeContext, FakeMemory, FakeOllama,  # noqa: F401
                                    brain, make_registry, text_reply, tool_reply)

GRADUATE = ("Do not forget that I graduate December 10th 2026 with an "
            "electrical engineering degree")
NOTED = "I have noted that, sir."


@pytest.fixture(autouse=True)
def _pin_the_greeting_hour(monkeypatch):
    """These tests are about the CLAIM GUARD, not about the greeting.

    brain.ground_greeting rightly rewrites a stale "Good evening" to the
    word the clock says (the bands live in jarvis/arc.py: morning 05-11,
    afternoon 12-16, evening 17-04). Four of the streamed cases below put
    a greeting in front of the honest line and then asserted the whole
    spoken list, so they were asserting a greeting they did not control:
    measured 2026-09-05, they passed 17:00-04:59 and failed 05:00-16:59,
    which is how a suite came to be green at 03:50 and red at 09:48 on one
    commit.

    The feature is NOT weakened, and no coverage is lost. ground_greeting
    still runs on every line in this file; it is simply told which hour it
    is, through arc.greeting_word -- the seam it already reads -- so it
    finds nothing stale and leaves the text alone. The rewrite itself was
    never this file's subject: it is owned by
    tests/test_found_stale_greeting.py, which pins `now=` and walks the
    real bands, and which is hour-independent already.
    """
    monkeypatch.setattr(arc_mod, "greeting_word", lambda now=None: "evening")


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
    # a pin (green before the pronoun rule below): the emphatic "don't
    # you <verb>" is an order too, and no '?' or wh-word makes it ask
    "Don't you forget that I graduate December 10th",
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


def test_only_the_remember_tool_backs_a_memory_claim():
    """Until 2026-09-12 no registry tool stored a fact and this set was
    empty; the `remember` tool (jarvis/tools/remember.py) is the one he
    ruled for and the only name here."""
    assert brain_mod.CLAIM_BACKERS["memory"] == frozenset({"remember"})
    assert brain_mod.claim_backed("I have noted that", ("remember",)) is True
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


# ------------------------------------------ the finisher's pass (2026-09-04)
# The author was stopped before reporting. Each item below was checked
# against the four commits on the FakeOllama harness; what follows is
# what was missing or wrong.

# (3) x (4): commit 7e0fccb taught guard() to return a released lead-in
# AND its sentence as a list; the kept-retry path from 12704c9 still
# treated guard()'s answer as one string. A retry that opens with an
# acknowledgement -- "Certainly, sir. I'm afraid I have no way to store
# that, sir." -- reached `" ".join(streamed_sentences)` with a list in it.
# The retry is kept only when it CLAIMS NOTHING, so there is no claim for
# a lead-in to be the yes to and nothing to hold it for: it is spoken as
# the plain path speaks it, in the model's order.
ACK_HONEST = "Certainly, sir. I'm afraid I have no way to store that, sir."


def test_a_kept_retry_that_opens_with_an_acknowledgement_is_spoken_whole(streamed):
    b, fake, record, streams = streamed
    streams.append(_chunks(f"{GREETING} I'll remember that, sir."))
    fake.replies = [text_reply(ACK_HONEST)]
    spoken = []
    tags = b._chat_sync("say hello to my family and remember that I graduate "
                        "December 10th", on_sentence=spoken.append)
    assert spoken == [GREETING, "Certainly, sir.",
                      "I'm afraid I have no way to store that, sir."]
    assert tags == [("STREAMED", "3"), ("SPEAK", f"{GREETING} {ACK_HONEST}")]
    assert record == []


def test_a_kept_retry_that_ends_with_an_acknowledgement_loses_nothing(streamed):
    """A trailing "Very well." was held for a sentence that never came
    and silently dropped; the plain path speaks it."""
    b, fake, record, streams = streamed
    streams.append(_chunks(f"{GREETING} I'll remember that, sir."))
    fake.replies = [text_reply(f"{HONEST} Very well.")]
    spoken = []
    tags = b._chat_sync("say hello to my family and remember that I graduate "
                        "December 10th", on_sentence=spoken.append)
    assert spoken == [GREETING, HONEST, "Very well."]
    assert tags == [("STREAMED", "3"), ("SPEAK", f"{GREETING} {HONEST} Very well.")]


# (1): _IMPERATIVE_DO_RX matched "do not" as a prefix, so "Do notes sync
# to my phone?" -- a question -- became an order and armed the guard on a
# question, the one thing is_question exists to prevent (a retry on a
# question can write). A word boundary ends the order.
@pytest.mark.parametrize("text", [
    "Do notes sync to my phone?",
    "Do notifications reach my phone",
    "Do notable events go in the briefing?",
])
def test_do_before_a_word_that_starts_with_not_is_still_a_question(text):
    assert is_question(text) is True, text


# (2), the refuter's probe d: the retry for a memory claim runs a real
# tool (get_time) and its render round claims the store afresh. The
# render is the reply -- the first would drop the time -- with the claim
# replaced by the memory line. Pinned on both paths; measured green on
# 7e0fccb before this section was written (a pin, not a fix).
def test_a_retry_that_runs_a_tool_and_reclaims_the_store_is_stripped(setup, caplog):
    b, fake, record = setup
    fake.replies = [text_reply(NOTED), tool_reply(("get_time", {})),
                    text_reply(TIME_AND_LIE)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(STORE)
    assert tags == [("SPEAK", f"It's five past four, sir. {MEMORY_LINE}")]
    assert record == [("get_time", None)]
    assert len(_warnings(caplog)) == 2


def test_a_streamed_retry_that_runs_a_tool_and_reclaims_speaks_the_time_then_the_line(streamed):
    b, fake, record, streams = streamed
    streams.append(_chunks(NOTED))                        # withheld, nothing spoken
    fake.replies = [tool_reply(("get_time", {}))]         # the plain retry: a tool
    streams.append(_chunks(TIME_AND_LIE))                 # its render round, streamed
    spoken = []
    tags = b._chat_sync(STORE, on_sentence=spoken.append)
    assert spoken == ["It's five past four, sir.", MEMORY_LINE]
    assert record == [("get_time", None)]
    assert tags == [("STREAMED", "2"), ("SPEAK", f"It's five past four, sir. {MEMORY_LINE}")]


# ------------------------------------ (6) an observation wearing the verb
# "I have noted that the lab has no listed duration" is an OBSERVATION in
# the store's clothes. _memory_claim_src takes one switch: strict flags
# every "noted/remember that ..." whatever follows; observation_exempt
# lets "that <clause>" through and keeps "that, sir" / "that." / "that
# for you". Both rules are measured here over 60 sentences in three
# groups -- STORE (a claim to have stored a fact: must flag), OBSERVATION
# (the judgement calls), PLAIN (ordinary English: must never flag) -- and
# the decision is pinned with its cost.
#
# The refuter's own 40-sentence list did not survive (the workflow's logs
# were empty by the time the finisher read them); the 40 below are every
# sentence its report names verbatim plus that sentence's family, and the
# 20 marked MINE are the finisher's. The four OBSERVATIONs it named are
# all here.
STORE_, OBS_, PLAIN_ = "store", "observation", "plain"
REFUTER_FAMILY = [
    # the lie, in every coat the report lists
    ("I have noted that, sir.", STORE_),
    ("I have noted that you graduate December 10th.", STORE_),   # the incident, fact spelled out
    ("I'll remember that, sir.", STORE_),
    ("I've saved that to memory.", STORE_),
    ("I'll keep that in mind, sir.", STORE_),
    ("I've made a note of that, sir.", STORE_),
    ("I won't forget that, sir.", STORE_),
    ("I'll make a note of that for you.", STORE_),
    ("I noted that already, sir.", STORE_),
    ("I have noted that for you, sir.", STORE_),
    # the four observations the report measured as FLAG
    ("I have noted that the lab has no listed duration.", OBS_),
    ("I've noted that it rains tomorrow.", OBS_),
    ("I will remember that meeting fondly, sir.", OBS_),
    ("I'll remember that face.", OBS_),
    # the legitimate lines it named, and their families
    ("I've saved you twenty minutes, sir.", PLAIN_),
    ("I've saved you some time, sir.", PLAIN_),
    ("I've saved you the trouble of a second trip.", PLAIN_),
    ("I've saved you a trip to the lab, sir.", PLAIN_),
    ("As I noted earlier, the lab has no listed duration.", PLAIN_),
    ("As I noted this morning, your flight is at nine.", PLAIN_),
    ("You noted that yourself last week, sir.", PLAIN_),
    ("It is noted in your calendar as a tentative hold.", PLAIN_),
    ("I remember that day well, sir.", PLAIN_),
    ("I can't remember that, sir; it was before my time.", PLAIN_),
    ("I haven't noted anything about your graduation, sir.", PLAIN_),
    ("Shall I remember that for you, sir?", PLAIN_),
    ("Would you like me to keep that in mind?", PLAIN_),
    ("I'll keep it short, sir.", PLAIN_),
    ("I'll remember this evening for a long time, sir.", PLAIN_),
    ("I'm afraid I have no way to store that, sir; say remember that and I shall.", PLAIN_),
    ("Nothing is stored in my memory about that, sir.", PLAIN_),
    ("I don't remember that, sir.", PLAIN_),
    ("Just as I noted yesterday, the dentist is at ten.", PLAIN_),
    ("Your calendar notes that the lab is booked until four.", PLAIN_),
    ("I remember it well: you said December 10th.", PLAIN_),
    ("That is noted in your journal, sir, not in my memory.", PLAIN_),
    ("I'll keep that short and to the point, sir.", PLAIN_),
    ("I remember that you prefer tea, sir.", PLAIN_),
    ("You'll remember that the lab closes early on Fridays.", PLAIN_),
    ("I noted nothing unusual in the logs, sir.", PLAIN_),
]
MINE = [
    # stores about a THIRD PARTY or a THING: what "Do remember that Heather
    # prefers email" earns as a reply. Only "that you ..." would be caught
    # by a rule that exempts "that <clause>", and these are not about him.
    ("I've noted that Heather prefers email, sir.", STORE_),
    ("I have noted that your locker code is 4412.", STORE_),
    ("I'll remember that the lab moved to room 049.", STORE_),
    ("I'll remember that you prefer the window seat.", STORE_),
    ("I've committed that to memory: December 10th.", STORE_),
    # observations of the same family as the report's four
    ("I've noted that the file is empty, sir.", OBS_),
    ("I have noted that nothing is on your calendar today.", OBS_),
    ("I've noted that the meeting overlaps your dentist appointment.", OBS_),
    ("I'll remember that one, sir; it was a good joke.", OBS_),
    ("I have noted that the printer is offline again.", OBS_),
    ("I'm noting that the forecast has changed since this morning.", OBS_),
    ("I've remembered that the shop shuts at six on Sundays.", OBS_),
    ("I have noted that the two of them share a birthday.", OBS_),
    # plain
    ("I keep that in mind whenever I plan your mornings, sir.", PLAIN_),
    ("Do you want me to make a note of that?", PLAIN_),
    ("I could make a note of that if you like, sir.", PLAIN_),
    ("I have no memory of that, sir.", PLAIN_),
    ("It was noted at the time, sir, by your supervisor.", PLAIN_),
    ("I'd remember that if I could, sir.", PLAIN_),
    ("Noted in passing: the kettle is on.", PLAIN_),
]
CORPUS = REFUTER_FAMILY + MINE


def _flagged_by(observation_exempt):
    """The corpus sentences the table flags under one rule, by group."""
    import re
    src = brain_mod._memory_claim_src(observation_exempt=observation_exempt)
    # the memory branch alone, in the table's own position and with the
    # same three vetoes _sentence_claim applies
    rx = re.compile(r"\b(?:" + src + r")", re.I)
    out = {STORE_: [], OBS_: [], PLAIN_: []}
    for sent, group in CORPUS:
        if brain_mod._CLAIM_HEDGE_RX.search(sent):
            continue
        for m in rx.finditer(sent):
            if brain_mod._claim_negated(sent[:m.start()]):
                continue
            if brain_mod._CLAIM_REPORTED_RX.search(sent[:m.start()]):
                continue
            if brain_mod._CLAIM_IDIOM_RX.match(sent[m.end():]):
                continue
            out[group].append(sent)
            break
    return out


def test_the_corpus_is_the_sixty_it_says_it_is():
    assert len(REFUTER_FAMILY) == 40 and len(MINE) == 20
    assert len({s for s, _ in CORPUS}) == 60
    assert [g for _, g in CORPUS].count(STORE_) == 15
    assert [g for _, g in CORPUS].count(OBS_) == 12
    assert [g for _, g in CORPUS].count(PLAIN_) == 33


def test_strict_is_the_rule_compiled_and_this_is_what_it_costs():
    """MEASURED over the 60: strict flags 15/15 stores and 0/33 plain,
    at the price of 12/12 observations (a retry each, kept when honest).
    observation_exempt flags 0/33 plain too, and 0/12 observations --
    but only 10/15 stores: it lets "I have noted that you graduate
    December 10th" (the incident with its fact spelled out) and every
    store about a third party or a thing walk through, five in all. A
    store's object is any fact he gives, so no rule on the WORDS after
    "that" can tell a store from an observation; the retry can. Strict
    stands. (The finisher guessed 9/15 before measuring; it is 10.)"""
    strict, exempt = _flagged_by(False), _flagged_by(True)
    assert brain_mod._MEMORY_CLAIM_SRC == brain_mod._memory_claim_src(observation_exempt=False)
    # the compiled table agrees with the strict branch measured alone
    assert sorted(s for s, g in CORPUS if brain_mod.unbacked_claim(s)) == \
        sorted(strict[STORE_] + strict[OBS_] + strict[PLAIN_])
    assert (len(strict[STORE_]), len(strict[OBS_]), len(strict[PLAIN_])) == (15, 12, 0)
    assert (len(exempt[STORE_]), len(exempt[OBS_]), len(exempt[PLAIN_])) == (10, 0, 0)
    missed = [s for s, g in CORPUS if g == STORE_ and s not in exempt[STORE_]]
    assert "I have noted that you graduate December 10th." in missed
    assert "I've noted that Heather prefers email, sir." in missed
    assert len(missed) == 5


# ------------------------------------------ the finisher's pass (2026-09-04, r2)
# The round-2b verifier reproduced two holes on bd978cf, on this harness.
# Each is pinned below by the probe that found it, and each pin was run
# red against bd978cf's sources before the fix was written.

# BLOCKER 1 (M15). _IMPERATIVE_DO_RX exempted only "don't you", so "Don't
# I have milk on my list already?" -- a QUESTION about the list -- became
# an order: the guard armed, the retry called notes(add, milk) and he
# heard "Noted, sir." The question performed the write it asked about,
# the 09-02 bug the question gate exists to prevent. On v3 he heard the
# answer and nothing was written.
#
# The rule, finished: a negative auxiliary before a SUBJECT PRONOUN asks
# ("don't I", "didn't you", "isn't it", "haven't we"), with or without the
# '?' the voice path drops; only "don't you <verb>" keeps an imperative
# reading ("don't you forget") and stays with the '?' rule. An imperative
# "do" before a verb orders. Twenty of each, measured on is_question and
# -- the questions -- on the brain, where the answer is what matters.
QUESTIONS_20 = [
    "Don't I have milk on my list already?",
    "Don't I have a meeting at ten",                  # no '?': the voice path
    "Do I have anything on tomorrow?",
    "Do I have milk on my list",
    "Don't you think it's late?",
    "Don't you remember that I graduate December 10th?",
    "Do we have a dentist appointment tomorrow?",
    "Do we have milk",
    "Didn't I ask you to add milk?",
    "Didn't I tell you about the lab move",
    "Didn't you set my timer for ten minutes",
    "Don't we have a dentist appointment tomorrow?",
    "Don't they close at six on Sundays?",
    "Doesn't it rain tomorrow",
    "Isn't it late",
    "Aren't we meeting at ten?",
    "Haven't I got a reminder set for five",
    "Jarvis, don't I have milk on my list already",
    "Do you know what time it is?",
    "Do notes sync to my phone?",
]
ORDERS_20 = [
    GRADUATE,
    "Don't forget that I graduate December 10th 2026",
    "Do remember that Heather prefers email",
    "Do keep in mind that the lab moved to room 049",
    "Do note that my locker code is 4412",
    "Do make a note that I graduate December 10th",
    "Do bear in mind that I graduate December 10th",
    "please do not forget that I graduate December 10th",
    "Please, do remember that I graduate December 10th",
    "Jarvis, do not forget that I graduate December 10th",
    "Jarvis, don't forget that I graduate December 10th",
    "Don't you forget that I graduate December 10th",   # the emphatic imperative
    "Don't ever forget that I graduate December 10th",
    "Don't forget to add milk to my list",
    "Do not add milk to my list twice",
    "Don't add milk to my list",
    "Do not set a reminder for five",
    "Don't set my alarm for six",
    "Do not tell Heather about the surprise",
    "Don't remember that, it was a joke",
]


def test_twenty_of_each():
    assert len(set(QUESTIONS_20)) == 20 and len(set(ORDERS_20)) == 20
    assert not set(QUESTIONS_20) & set(ORDERS_20)


@pytest.mark.parametrize("text", QUESTIONS_20)
def test_a_question_in_each_of_twenty_shapes_asks(text):
    assert is_question(text) is True, text


@pytest.mark.parametrize("text", ORDERS_20)
def test_an_order_in_each_of_twenty_shapes_orders(text):
    assert is_question(text) is False, text


YOU_DO = "You do, sir. I've added it to your list already."


@pytest.mark.parametrize("asked", QUESTIONS_20)
def test_a_question_in_each_of_twenty_shapes_is_never_acted_on(setup, caplog, asked):
    """He asked. The model's answer claims an add it never made; that is
    a possibly-wrong ANSWER, not an order, and the retry that would make
    it true must never run -- so nothing is written and no round is
    spent, whatever shape the question took."""
    b, fake, record = setup
    fake.replies = [text_reply(YOU_DO),
                    tool_reply(("notes", {"action": "add", "text": "milk"}))]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(asked)
    assert record == [], asked
    assert tags == [("SPEAK", YOU_DO)]
    assert len(fake.chat_payloads()) == 1
    assert _warnings(caplog) == []


@pytest.mark.parametrize("ordered", ORDERS_20)
def test_an_order_in_each_of_twenty_shapes_arms_the_guard(setup, caplog, ordered):
    """The mirror: every order arms the guard, so "I have noted that,
    sir." twice over is never what he hears."""
    b, fake, record = setup
    fake.replies = [text_reply(NOTED), text_reply(NOTED)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(ordered)
    spoken = dict(tags)["SPEAK"]
    assert "noted that" not in spoken.lower(), (ordered, spoken)
    assert record == []
    assert _warnings(caplog), "the guard never armed"
    assert len(fake.chat_payloads()) == 2


# A question that also asks for a store arms the guard for MEMORY claims
# only (test_a_question_that_asks_for_a_store_judges_memory_claims_only),
# and its retry still had the tools: "Don't you remember that I graduate
# December 10th?" answered "Of course, sir. I'll remember that." earned a
# retry that could call notes(add, ...) -- the guard itself writing on a
# question. The retry on a question is offered NO tools: it may answer in
# words (kept when honest, replaced when it claims again), and a tool
# call it makes anyway is dropped, as the render round drops them. An
# order's retry keeps its tools: that is where "add milk" gets done.
REMEMBER_Q = "Don't you remember that I graduate December 10th?"


def test_the_retry_on_a_question_is_offered_no_tools_and_cannot_write(setup, caplog):
    b, fake, record = setup
    assert is_question(REMEMBER_Q)
    fake.replies = [text_reply(OF_COURSE),
                    tool_reply(("notes", {"action": "add",
                                          "text": "I graduate December 10th"}))]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(REMEMBER_Q)
    assert record == []
    assert tags == [("SPEAK", MEMORY_LINE)]
    p1, p2 = fake.chat_payloads()
    assert p1.get("tools")
    assert "tools" not in p2
    # On a QUESTION the retry has no tools, so the nudge does not order a
    # remember call it cannot make; it asks for the honest answer.
    assert p2["messages"][-1]["content"] == brain_mod.UNBACKED_MEMORY_NUDGE_QUESTION


def test_the_retry_on_a_question_may_still_answer_in_words(setup):
    b, fake, record = setup
    fake.replies = [text_reply(OF_COURSE), text_reply(HONEST)]
    assert b._chat_sync(REMEMBER_Q) == [("SPEAK", HONEST)]
    assert "tools" not in fake.chat_payloads()[1]
    assert record == []


def test_streamed_the_retry_on_a_question_that_tries_to_write_speaks_the_line(streamed):
    b, fake, record, streams = streamed
    streams.append(_chunks(OF_COURSE))                    # withheld, nothing spoken
    fake.replies = [tool_reply(("notes", {"action": "add",
                                          "text": "I graduate December 10th"}))]
    spoken = []
    tags = b._chat_sync(REMEMBER_Q, on_sentence=spoken.append)
    assert spoken == [MEMORY_LINE]
    assert tags == [("STREAMED", "1"), ("SPEAK", MEMORY_LINE)]
    assert record == []


def test_the_retry_on_an_order_keeps_its_tools(setup):
    """A pin, not a change: the order's retry is where the work gets
    done, so its tools stay."""
    b, fake, record = setup
    fake.replies = [text_reply("I've added milk to your list, sir."),
                    tool_reply(("notes", {"action": "add", "text": "milk"}))]
    assert b._chat_sync("Add milk to my list") == [("SPEAK", "Noted, sir.")]
    assert record == [("notes", "add", "milk")]
    assert fake.chat_payloads()[1].get("tools")


# BLOCKER 2 (M05 / M05b). "Check the weather, set a reminder for five and
# remember that I graduate December 10th": get_weather and set_reminder
# ran, the render reported both AND claimed the store, twice. brain.py
# stripped the first reply with strip_unbacked_claims(source, cap,
# kinds=...) and no ``ran``, so "I've set your reminder for five" --
# BACKED by the set_reminder that ran, judged so when the claim was
# found -- was unbacked at strip time and became "I couldn't do that
# part, sir." beside the memory line; with set_reminder's own speak=
# line appended after, he heard the reminder denied and confirmed in one
# reply (measured: "Seventy-two and cloudy, sir. I couldn't do that part,
# sir. I can't store that from here, sir -- say 'remember that ...' and
# I will. Reminder set for five o'clock, sir.", three model rounds).
# ONE coherent reply: what ran is confirmed, the memory clause gets the
# authored line exactly once.
ORDER_THREE = ("Check the weather, set a reminder for five and remember that "
               "I graduate December 10th")
RENDER_THREE = ("Seventy-two and cloudy, sir. I've set your reminder for five, "
                "sir. I'll remember that.")
REMINDER_KEPT = "Seventy-two and cloudy, sir. I've set your reminder for five, sir."
REMINDER_LINE = "Reminder set for five o'clock, sir."


def _add_reminder(record, speak=None):
    from jarvis.tools.registry import ToolResult, ToolSpec

    def set_reminder(when="", **_):
        record.append(("set_reminder", when))
        return ToolResult(text=f"reminder set for {when}", speak=speak)
    brain_mod._REGISTRY.register_many([
        ToolSpec("set_reminder", "Set a reminder.",
                 {"type": "object", "properties": {"when": {"type": "string"}}},
                 set_reminder)])


THREE_RAN = [("get_weather", "now", None), ("set_reminder", "five")]


def _once(spoken):
    """The memory line exactly once, the action denial never."""
    assert spoken.count(MEMORY_LINE) == 1, spoken
    assert brain_mod.UNBACKED_LINE not in spoken, spoken


def test_a_reminder_that_ran_is_not_denied_beside_a_memory_claim(setup, caplog):
    b, fake, record = setup
    _add_reminder(record)
    fake.replies = [tool_reply(("get_weather", {"when": "now"}),
                               ("set_reminder", {"when": "five"})),
                    text_reply(RENDER_THREE), text_reply(RENDER_THREE)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(ORDER_THREE)
    assert tags == [("SPEAK", f"{REMINDER_KEPT} {MEMORY_LINE}")]
    _once(dict(tags)["SPEAK"])
    assert record == THREE_RAN
    assert len(fake.chat_payloads()) == 3                 # tools, render, retry
    assert _warnings(caplog) == [
        "brain: unbacked memory claim \"I'll remember that\" (get_weather, "
        "set_reminder ran; nothing that ran stores a fact)",
        "brain: unbacked action claim stands after the retry (no tool ran); "
        "replacing it"]


def test_a_reminder_with_its_own_line_is_not_denied_beside_a_memory_claim(setup):
    """The same turn with set_reminder's speak= line: the line is spoken
    after the render as before; what must not be there is a denial."""
    b, fake, record = setup
    _add_reminder(record, speak=REMINDER_LINE)
    fake.replies = [tool_reply(("get_weather", {"when": "now"}),
                               ("set_reminder", {"when": "five"})),
                    text_reply(RENDER_THREE), text_reply(RENDER_THREE)]
    spoken = dict(b._chat_sync(ORDER_THREE))["SPEAK"]
    _once(spoken)
    assert spoken.startswith(f"{REMINDER_KEPT} {MEMORY_LINE}"), spoken
    assert spoken.endswith(REMINDER_LINE), spoken
    assert record == THREE_RAN
    assert len(fake.chat_payloads()) == 3


def _tool_chunks(*calls):
    return [{"message": {"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": n, "arguments": a}}
                                        for n, a in calls]},
             "done": True, "load_duration": 0}]


def test_streamed_a_backed_action_claim_is_not_unsaid_by_the_line(streamed):
    """On the stream the reminder sentence went out as it landed (backed
    by set_reminder, guard() let it through); the memory claim was
    withheld. The round-end line must be the memory line ALONE -- an
    "I couldn't do that part" after a confirmation already spoken would
    unsay it."""
    b, fake, record, streams = streamed
    _add_reminder(record)
    streams.append(_tool_chunks(("get_weather", {"when": "now"}),
                                ("set_reminder", {"when": "five"})))
    streams.append(_chunks(RENDER_THREE))
    fake.replies = [text_reply(RENDER_THREE)]          # the plain retry
    spoken = []
    tags = b._chat_sync(ORDER_THREE, on_sentence=spoken.append)
    assert spoken == ["Seventy-two and cloudy, sir.",
                      "I've set your reminder for five, sir.", MEMORY_LINE]
    assert tags == [("STREAMED", "3"), ("SPEAK", f"{REMINDER_KEPT} {MEMORY_LINE}")]
    _once(dict(tags)["SPEAK"])
    assert record == THREE_RAN


# ------------------------------------ the fifth window's fix (2026-09-05)
# The round-4 reviewer and verdict each measured, on this harness, that
# on the order "Remember that I graduate December 10th 2026" the reply
# "No problem, I have noted that, sir." was spoken VERBATIM with nothing
# stored and zero warnings -- and so were five more lead-in shapes and the
# streamed form. Cause, confirmed in code: _sentence_claim vetoed a claim
# on ANY negation word earlier in the SENTENCE (sent[:m.start()]), so the
# "No" of "No problem," was read as governing a claim two clauses away.
# The same line is on jarvis-v3, so this is a fix-forward, not a
# regression; the persona's own few-shot answers "Not at all, sir.", so
# the shape is likely in the wild.
#
# The rule, finished: a negation vetoes a claim only when it governs the
# claim's own clause -- the text after the last clause boundary (, ; :
# dash) -- and a lead-in idiom that wears a negation word and governs
# nothing ("no problem", "not to worry", "never fear", "I can't forget
# that") is no negation of what follows it, comma or no comma. A real
# veto ("I have not noted", "I haven't saved", "Nothing has been added
# to your list") sits in the claim's own clause and still holds.
REMEMBER_ORDER = "Remember that I graduate December 10th 2026"
LEAD_IN_SIX = [
    "No problem, I have noted that, sir.",
    "Not a problem, sir, I've saved that to memory.",
    "No worries, I'll remember that, sir.",
    "Not at all, sir; I've made a note of that.",
    "I can't store that, sir, but I've noted it.",
    "I don't have a memory tool as such, but I'll keep that in mind, sir.",
]


def test_the_order_that_was_measured_is_an_order():
    assert is_question(REMEMBER_ORDER) is False


@pytest.mark.parametrize("line", LEAD_IN_SIX + [
    # the brief's other lead-ins, each before a real claim
    "Not a problem — I've saved that to memory.",       # a dash is a boundary
    "Don't worry, sir, I've noted that.",
    "Not to worry, I have noted that, sir.",
    "Never fear, sir, I'll remember that.",
    "I can't forget that, sir — I've made a note of it.",
    "No trouble at all, sir, I'll keep that in mind.",
    "Not at all: I've committed that to memory, sir.",
    # the same lead-ins with the comma the voice path never hears
    "No problem sir I've noted that.",
    "Don't worry sir I'll remember that.",
    "Never fear I have noted that, sir.",
    # a bare "No," is a lead-in too
    "No, sir, I've noted that.",
])
def test_a_lead_in_negation_does_not_veto_the_claim(line):
    assert brain_mod.unbacked_claim(line), line
    assert brain_mod.claim_kind(brain_mod.unbacked_claim(line)) == "memory", line


@pytest.mark.parametrize("line", [
    # the brief's three
    "I can't store that from here, sir.",
    "I haven't saved anything, sir.",
    "I did not note it, sir.",
    # the negation in the claim's own clause: still a veto
    "I have not noted that, sir.",
    "I won't remember that, sir.",
    "Nothing has been added to your list, sir.",
    "Nothing is added to your calendar yet.",
    "No music is on, sir.",
    "I don't think the music is on, sir.",
    "It's not as if I've added milk, sir.",
    # a hedge that wears "not at all" mid-clause is not a lead-in
    "I'm not at all sure I've added milk, sir.",
    # the negation and the claim share a clause AFTER a lead-in
    "No problem, sir, but I haven't noted anything yet.",
    "Not to worry, nothing has been added to your list.",
])
def test_a_negation_that_governs_the_claim_still_vetoes_it(line):
    assert brain_mod.unbacked_claim(line) is None, line


@pytest.mark.parametrize("line", LEAD_IN_SIX)
def test_a_lead_in_negation_reply_earns_the_memory_line(setup, caplog, line):
    """The six the reviewer measured verbatim: on the order, each is now
    caught, retried, and -- the retry claiming again -- replaced by the
    memory line alone; nothing stored."""
    b, fake, record = setup
    fake.replies = [text_reply(line), text_reply(line)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(REMEMBER_ORDER)
    assert tags == [("SPEAK", MEMORY_LINE)], line
    assert record == []
    assert len(_warnings(caplog)) == 2, line              # the claim, then it stands
    assert len(fake.chat_payloads()) == 2                 # the one retry


def test_a_streamed_lead_in_negation_reply_earns_the_memory_line(streamed, caplog):
    """The streamed form: the sentence is withheld on the stream (it is
    one sentence, lead-in and claim together) and he hears the line."""
    b, fake, record, streams = streamed
    streams.append(_chunks(LEAD_IN_SIX[0]))
    fake.replies = [text_reply(LEAD_IN_SIX[0])]
    spoken = []
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(REMEMBER_ORDER, on_sentence=spoken.append)
    assert spoken == [MEMORY_LINE]
    assert tags == [("STREAMED", "1"), ("SPEAK", MEMORY_LINE)]
    assert record == [] and len(_warnings(caplog)) == 2


def test_a_lead_in_negation_reply_still_yields_to_an_honest_retry(setup):
    b, fake, record = setup
    fake.replies = [text_reply(LEAD_IN_SIX[0]), text_reply(HONEST)]
    assert b._chat_sync(REMEMBER_ORDER) == [("SPEAK", HONEST)]
    assert record == []


STRAY_HONEST = ("I'm afraid I have no way to store that, sir, but I'll keep "
                "that in mind.")


def test_an_honest_retry_with_a_stray_claim_is_not_spoken(setup, caplog):
    """The verdict's H2: the retry's "no way" vetoed its own trailing
    "I'll keep that in mind" and the stray claim was spoken. It is a
    second claim; the first reply, stripped, is what he hears."""
    b, fake, record = setup
    fake.replies = [text_reply(NOTED), text_reply(STRAY_HONEST)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(STORE)
    assert tags == [("SPEAK", MEMORY_LINE)]
    assert record == [] and len(_warnings(caplog)) == 2
    assert brain_mod.unbacked_claim(STRAY_HONEST) == "I'll keep that in mind"


RENDER_THREE_LEAD_IN = ("Seventy-two and cloudy, sir. I've set your reminder for "
                        "five, sir. No worries, I'll remember that, sir.")


def test_the_three_round_count_holds_with_a_lead_in_negation(setup, caplog):
    """The multi-tool turn with the memory clause worn as a lead-in shape:
    memory line once, the action denial never, both tools confirmed."""
    b, fake, record = setup
    _add_reminder(record)
    fake.replies = [tool_reply(("get_weather", {"when": "now"}),
                               ("set_reminder", {"when": "five"})),
                    text_reply(RENDER_THREE_LEAD_IN),
                    text_reply(RENDER_THREE_LEAD_IN)]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync(ORDER_THREE)
    spoken = dict(tags)["SPEAK"]
    assert spoken == f"{REMINDER_KEPT} {MEMORY_LINE}"
    _once(spoken)
    assert record == THREE_RAN
    assert len(fake.chat_payloads()) == 3
    assert len(_warnings(caplog)) == 2


# ---------------------------------- the remember tool's refusal (09-12 attack)
def test_a_refused_remember_call_does_not_back_a_memory_claim(setup, caplog):
    """CLAIM_BACKERS['memory'] = {'remember'} named the tool; the backer
    list was every tool that RAN, ok or not. So a refused remember (a
    question, another tool's job, a failed store) followed by 'I have
    noted that, sir' was spoken as backed -- the exact lie the guard was
    built from. Only a run that STORED backs the claim."""
    from jarvis.tools import remember as rt
    from types import SimpleNamespace
    b, fake, record = setup
    reg = brain_mod._REGISTRY
    reg.register_many(rt.make_tools(None, SimpleNamespace(memory=None)))   # every call refuses
    fake.replies = [tool_reply(("remember", {"fact": "remind me to call mom at 5"})),
                    text_reply("I have noted that, sir."),
                    text_reply("I can't do that from here, sir.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("Put it in your memory that I need to call mom at 5")
    spoken = dict(tags)["SPEAK"]
    assert "noted" not in spoken.lower(), spoken
    assert _warnings(caplog), "the refused remember backed the claim"
    assert len(fake.chat_payloads()) == 3                      # the one retry


def test_a_stored_remember_call_backs_the_claim(setup, caplog):
    from jarvis.tools import remember as rt
    from jarvis.memory import JarvisMemory
    from types import SimpleNamespace
    import tempfile
    from pathlib import Path
    b, fake, record = setup
    tmp = Path(tempfile.mkdtemp())
    memory = JarvisMemory(memory_dir=tmp / "mem", legacy_dir=tmp / "legacy", semantic=False)
    brain_mod._REGISTRY.register_many(rt.make_tools(None, SimpleNamespace(memory=memory)))
    fake.replies = [tool_reply(("remember", {"fact": "I graduate December 10th 2026"})),
                    text_reply("I have noted that, sir.")]
    with caplog.at_level("INFO", logger="jarvis.brain"):
        tags = b._chat_sync("Remember that I graduate December 10th 2026")
    spoken = dict(tags)["SPEAK"]
    assert spoken.startswith("Noted, sir:")                    # the authored line
    assert not _warnings(caplog)
    assert memory.recall("graduate")
