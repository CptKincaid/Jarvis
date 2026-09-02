"""Regression tests: "sir" was said to Hunter's FAMILY.

LIVE 2026-09-02 14:29:24, /tmp/vss_voice/jarvis.log line 6520. Hunter said
"Say hello to my family." and the room heard:

    Good evening, Ali and Heather; I do hope you're both having a lovely
    afternoon, sir.

The greeting is addressed to Ali and Heather -- "you're BOTH" -- and then
signs off to a third person who is not in the sentence. The persona is
unconditional about it: ``brain.VOICE_RULES`` says call him "sir" most of
the time, ``JARVIS_SYSTEM`` closes with "call him sir", and the formal
register says "every time". None of them knows the audience.

THE RULE, deliberately narrow. "sir" is the character and it must not
start thinning out of ordinary replies, so BOTH gates have to hold before
a single vocative is dropped:

  1. Hunter's own words asked for a RELAY -- "say/tell/wish/greet ..." with
     an audience that is a person and is not himself. "tell me a joke",
     "what did you say to me" and "tell the truth" are not relays.
  2. The spoken line itself ADDRESSES that audience: it opens by greeting
     someone by name who is not Hunter.

Gate 2 is what keeps "Tell my professor I'll be late." -> "I'm afraid I
can't send messages, sir." intact: that sentence is spoken TO Hunter, and
it keeps its sir.

The removal is ``address.drop_addresses``, so it inherits every measured
rule in jarvis/address.py -- a possessive ("sir's coffee"), an honorific
before a name ("Sir Isaac Newton") and a "sir" inside a quotation are not
forms of address and are never touched.
"""
import pytest

from jarvis import address
from jarvis.brain import _finish_spoken, relay_request, strip_relay_address

FAMILY = ("Good evening, Ali and Heather; I do hope you're both having a "
          "lovely afternoon, sir.")


# ------------------------------------------------------------------ gate 1
@pytest.mark.parametrize("text", [
    "Say hello to my family.",
    "Say hi to my family and then give me my daily briefing.",
    "Say good night to Ali.",
    "Tell my brother I'll be late.",
    "Wish Heather a happy birthday.",
    "Greet the family for me.",
])
def test_relay_requests_are_recognised(text):
    assert relay_request(text) is True


@pytest.mark.parametrize("text", [
    "Tell me a joke.",
    "Tell me what's on my calendar.",
    "Say that again.",
    "What did you say to me?",
    "Tell the truth, was that a guess?",
    "Say hello.",
    "",
])
def test_ordinary_requests_are_not_relays(text):
    assert relay_request(text) is False


# ------------------------------------------------------------------ gate 2
def test_the_live_line_loses_its_sir():
    assert strip_relay_address(FAMILY, "Say hello to my family.") == (
        "Good evening, Ali and Heather; I do hope you're both having a "
        "lovely afternoon.")


def test_a_reply_spoken_to_hunter_keeps_its_sir():
    """The relay was requested, but this sentence answers HIM."""
    line = "I'm afraid I can't send messages, sir."
    assert strip_relay_address(line, "Tell my professor I'll be late.") == line


def test_an_ordinary_reply_keeps_its_sir():
    line = "Good afternoon, sir."
    assert strip_relay_address(line, "Hello Jarvis.") == line


def test_a_greeting_addressed_to_hunter_by_name_keeps_its_sir():
    line = "Good afternoon, Hunter; the build passed, sir."
    assert strip_relay_address(line, "Say hi to my family.") == line


def test_a_sentence_that_addresses_them_as_a_group_also_loses_it():
    """The other half of the live line -- "I do hope you're BOTH having a
    lovely afternoon". A plural second person is not Hunter on his own."""
    line = ("Good afternoon, Ali and Heather. I hope you're both keeping "
            "well, sir.")
    out = strip_relay_address(line, "Say hello to my family.")
    assert "sir" not in out.lower()
    assert out.startswith("Good afternoon, Ali and Heather.")


def test_a_compound_request_keeps_the_sir_in_the_half_meant_for_him():
    """LIVE 14:43:56: "say hi to my family and then give me my daily
    briefing" is a relay AND an order. The greeting is theirs, the
    briefing is his, and a rule that carried the audience forward from
    sentence one would have taken the sir out of the briefing."""
    line = ("Hello, Ali and Heather. It is overcast today with a high of "
            "98, sir.")
    out = strip_relay_address(
        line, "Say hi to my family and then give me my daily briefing.")
    assert out == line


def test_the_production_path_drops_it():
    spoken = _finish_spoken(FAMILY, "", "Say hello to my family.", 2)
    assert "sir" not in spoken.lower()
    assert "Ali and Heather" in spoken


# ----------------------------------------------------- the safety net below
@pytest.mark.parametrize("line", [
    "Sir Isaac Newton wrote the Principia.",
    "That is sir's coffee.",
    "Now playing Yes Sir, I Can Boogie.",
])
def test_a_sir_that_is_not_a_form_of_address_is_never_cut(line):
    assert address.drop_addresses(line) == line
    assert strip_relay_address(line, "Say hello to my family.") == line


# --------------------------------------------------------- while streaming
def test_a_streamed_relay_loses_it_sentence_by_sentence(monkeypatch):
    """The path that actually spoke the defect. Sentence one reaches TTS
    before sentence two exists (LIVE 14:43:59.323 vs 14:44:00.864, 1.5 s
    apart), so the audience has to be decided as each sentence lands."""
    from tests.test_streaming_replies import _brain, _chunks

    b = _brain(monkeypatch,
               _chunks("Hello, Ali and Heather. I hope you're both keeping "
                       "well, sir."))
    spoken = []
    b._chat_sync("Say hello to my family.", on_sentence=spoken.append)
    assert spoken == ["Hello, Ali and Heather.",
                      "I hope you're both keeping well."]


def test_a_streamed_reply_to_hunter_keeps_its_sir(monkeypatch):
    from tests.test_streaming_replies import _brain, _chunks

    b = _brain(monkeypatch, _chunks("The build passed, sir. Nothing to do."))
    spoken = []
    b._chat_sync("how did the build go", on_sentence=spoken.append)
    assert spoken == ["The build passed, sir.", "Nothing to do."]


# ----------------------------------------------------------------------
# Gate 2's name class -- the 2026-09-02 over-reach review
#
# _GREETS_BY_NAME_RX was compiled with a blanket re.I, which made its
# [A-Z][a-z]+ "name" class match ANY lowercase word: every sentence opening
# with a greeting word was read as third-party speech and lost Hunter's
# honorific. Measured captures at the time:
#
#   'Welcome back, sir.'                                who='back, sir'
#   'Hi there, sir.'                                    who='there, sir'
#   'hello has been added to tomorrow at 4:30 pm, sir.' who='has'
#
# The first is presence.WELCOME_LINE verbatim and the third is a real
# gemma4 line from jarvis.log.1 21:02:50.994, so this was not hypothetical
# vocabulary. The case-insensitivity is now scoped to the greeting words.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("line", [
    "Welcome back, sir.",
    "Welcome home, sir.",
    "Hello again, sir.",
    "Hi there, sir.",
    "Hey now, sir.",
    "Greetings from the calendar, sir.",
    "Hello there, sir; the tests are running.",
    "Hi, the session is open, sir.",
    "Hey, that's done, sir.",
    "Good morning, everything is quiet, sir.",
])
def test_a_greeting_followed_by_an_ordinary_word_is_not_a_name(line):
    """The word after "hello" has to look like a name, not merely exist."""
    for asked in ("Say hello to my family.", "Tell my brother I'll be late.",
                  "Tell Claude to run the tests."):
        assert strip_relay_address(line, asked) == line


def test_the_welcome_line_jarvis_really_says_keeps_its_sir():
    """presence.WELCOME_LINE is authored and spoken to HIM. It does not
    traverse this guard in production, but a model reply of the same shape
    does, and under the loose regex both lost their sir."""
    from jarvis.presence import WELCOME_LINE

    assert WELCOME_LINE == "Welcome back, sir."
    assert strip_relay_address(WELCOME_LINE, "Say hello to my family.") == \
        WELCOME_LINE


def test_a_real_logged_calendar_line_keeps_its_sir():
    """jarvis.log.1 21:02:50.994, a line gemma4 actually produced. Under
    the loose regex it captured who='has' and was judged third-party."""
    line = "hello has been added to tomorrow at 4:30 pm, sir."
    assert strip_relay_address(line, "Say hi to my family.") == line


def test_the_second_sentence_of_a_relay_reply_keeps_the_sir_that_is_his():
    """The shape a relay turn actually produces: the greeting is theirs,
    what follows is his. Sentence two opens with a greeting WORD and is
    still addressed to Hunter, which is the case the first version of this
    file never exercised."""
    line = ("Good afternoon, Ali and Heather. Welcome back, sir. You have "
            "three items today, sir.")
    out = strip_relay_address(
        line, "say hi to my family and then tell me what's on my calendar")
    assert out == ("Good afternoon, Ali and Heather. Welcome back, sir. "
                   "You have three items today, sir.")


# ------------------------------------------------- "you all" is not plural
@pytest.mark.parametrize("line", [
    "You are all caught up, sir.",
    "Are you all set for the day, sir?",
    "Thank you all the same, sir.",
    "Is that you all right, sir?",
    "I'll send you two reminders, sir.",
    "I'll get you all set up, sir.",
    "That leaves you two options, sir.",
])
def test_a_quantity_after_you_is_not_a_plural_second_person(line):
    """"you two reminders" is one listener and two reminders. Only a
    clause boundary or a verb behind the quantity makes it an audience."""
    assert strip_relay_address(line, "Tell my brother I'll be late.") == line


@pytest.mark.parametrize("line", [
    "I do hope you're both having a lovely day, sir.",
    "I hope you are both well, sir.",
    "Both of you are very welcome, sir.",
    "You two are expected at seven, sir.",
    "I hope you both have a pleasant evening, sir.",
])
def test_a_genuine_plural_second_person_still_loses_it(line):
    out = strip_relay_address(line, "Say hello to my family.")
    assert "sir" not in out.lower()
