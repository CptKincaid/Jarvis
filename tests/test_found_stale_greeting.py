"""Regression tests: a greeting REPLAYED out of memory, seven hours stale.

LIVE 2026-09-02 14:29:24, /tmp/vss_voice/jarvis.log line 6520. "Say hello
to my family." at 2:29 in the afternoon was answered:

    Good evening, Ali and Heather; I do hope you're both having a lovely
    afternoon, sir.

Internally contradictory, and "Good evening" is wrong by six hours. The
arc had flipped seven seconds earlier (14:29:07 "arc: night -> afternoon")
and ``context.format_for_prompt`` had already put "Current time: 02:29 PM"
in the model's background, so nothing it needed was missing.

THE MECHANISM, verified rather than assumed. ``Memory.save_session`` files
the previous session verbatim and ``format_sessions_for_prompt`` renders
the last three into every user turn, each truncated to 100 characters.
~/.aiws_trainer/jarvis_memory/sessions.json, read on 2026-09-02:

  2026-09-01T20:55:08  You: Say hello to my family. / Jarvis: Good evening,
                       Ali and Heather; I do hope you're both having a
  2026-09-01T20:56:22  You: Say hi to my family. / Jarvis: Good evening,
                       Ali and Heather; I do hope you're both having a lo

So the model was handed its own opening for THIS EXACT REQUEST, twice,
cut off mid-clause, and it did the obvious thing: copied the opening and
finished the tail against the clock it could see ("a lovely afternoon").
That is also why the defect is intermittent -- at 14:43 the request was
worded differently, took the tool path, and came out "Good afternoon".

THE FIX: the time-of-day word is GROUNDED, not recalled. It comes from
the wall clock at the moment of speaking, on the same bands
``commander._greeting_line`` has always used (5-12 morning, 12-17
afternoon, else evening) -- so the two can never disagree, and a recalled
opening is corrected before it is spoken.

Grounded on the CLOCK and not on ``arc.phase()``, deliberately. The arc is
forced to "night" whenever the house is empty or hushed, and the same log
has it at 14:06:07 -- "arc: afternoon -> night (forced by you're out)". A
greeting keyed off the phase would have said "Good night" at six minutes
past two, which is the very bug being fixed. ``arc.greeting_word`` is
therefore a pure function of the clock and is documented as such.
"""
from datetime import datetime

import pytest

from jarvis import arc
from jarvis.brain import _finish_spoken, ground_greeting
from jarvis.commander import COURTESY_REPLIES, _greeting_line

STALE = ("Good evening, Ali and Heather; I do hope you're both having a "
         "lovely afternoon, sir.")
AT_1429 = datetime(2026, 9, 2, 14, 29)


def test_the_live_line_is_corrected():
    assert ground_greeting(STALE, now=AT_1429) == (
        "Good afternoon, Ali and Heather; I do hope you're both having a "
        "lovely afternoon, sir.")


@pytest.mark.parametrize("hour,word", [
    (5, "morning"), (9, "morning"), (11, "morning"),
    (12, "afternoon"), (14, "afternoon"), (16, "afternoon"),
    (17, "evening"), (20, "evening"), (23, "evening"), (2, "evening"),
])
def test_the_word_follows_the_clock(hour, word):
    now = datetime(2026, 9, 2, hour, 0)
    assert arc.greeting_word(now) == word
    assert ground_greeting("Good morning, sir.", now=now) == \
        f"Good {word}, sir."


def test_the_bands_are_the_ones_the_courtesy_reply_already_used():
    """One rule, two callers: commander's canned greeting and the guard on
    the model's must never disagree about what hour it is."""
    for hour in range(24):
        now = datetime(2026, 9, 2, hour, 0)
        expected = COURTESY_REPLIES["greeting"][arc.greeting_index(now)]
        assert _greeting_line(now) == expected
        assert arc.greeting_word(now) in expected


def test_a_correct_greeting_is_left_alone():
    assert ground_greeting("Good afternoon, sir.", now=AT_1429) == \
        "Good afternoon, sir."


def test_good_night_is_a_sign_off_and_is_never_rewritten():
    """"Good night, sir" is right whenever he says it -- 9 pm or 3 am."""
    line = "Good night, sir. I'll be here."
    assert ground_greeting(line, now=datetime(2026, 9, 2, 3, 0)) == line


def test_a_greeting_hunter_used_himself_is_not_contradicted():
    """He said it; Jarvis echoing him is politeness, not a stale memory."""
    assert ground_greeting("Good evening, sir.", "Good evening, Jarvis.",
                           now=AT_1429) == "Good evening, sir."


def test_only_a_sentence_opening_is_grounded():
    """A greeting he is being asked to pass on later is not a claim about
    the present hour."""
    line = "I'll wish them a good evening later, sir."
    assert ground_greeting(line, now=AT_1429) == line


def test_the_production_path_grounds_it():
    spoken = _finish_spoken(STALE, "", "Say hello to my family.", 2)
    assert spoken.lower().startswith("good afternoon") or \
        spoken.lower().startswith(f"good {arc.greeting_word()}")


# ------------------------------------------------- the source of the parrot
def test_previous_sessions_are_labelled_as_past_and_not_to_be_reused():
    """The block that fed the model its own old opening. It stays -- the
    recall is the point -- but it now says what it is, so the wording is
    not offered as a draft."""
    from jarvis.memory import JarvisMemory

    mem = JarvisMemory.__new__(JarvisMemory)
    mem._sessions = [{"time": "2026-09-01T20:55:08",
                      "summary": "You: Say hello to my family. / Jarvis: "
                                 "Good evening, Ali and Heather;"}]
    block = mem.format_sessions_for_prompt()
    assert "2026-09-01" in block
    assert "never reuse" in block.lower()


def test_a_title_that_begins_with_a_greeting_is_not_rewritten():
    """"Good Morning America" is a programme, not a claim about the hour."""
    line = "Good Morning America is at nine, sir."
    assert ground_greeting(line, now=AT_1429) == line


# ----------------------------------------------------------------------
# The rest of the sentence -- the 2026-09-02 over-reach review
#
# Grounding only the opening WORD leaves a recalled clause contradicting
# itself instead of the clock, which is the same shape of defect the fix
# was written to remove. On the real logged line, replayed at 11:59:
#
#   IN : "Good evening, Ali and Heather; I do hope you're both having a
#         pleasant evening."
#   OUT: "Good morning, Ali and Heather; I do hope you're both having a
#         pleasant evening."
#
# The live 14:29 line was the lucky case: its tail already said
# "afternoon", so grounding the opening happened to finish the sentence.
# ----------------------------------------------------------------------
AT_1159 = datetime(2026, 9, 2, 11, 59)


def test_a_recalled_well_wish_is_grounded_with_its_greeting():
    line = ("Good evening, Ali and Heather; I do hope you're both having a "
            "pleasant evening.")
    assert ground_greeting(line, now=AT_1159) == (
        "Good morning, Ali and Heather; I do hope you're both having a "
        "pleasant morning.")


def test_the_live_line_keeps_the_tail_that_was_already_right():
    """It said "a lovely afternoon" at 2:29 pm, and that was correct."""
    assert ground_greeting(STALE, now=AT_1429) == (
        "Good afternoon, Ali and Heather; I do hope you're both having a "
        "lovely afternoon, sir.")


@pytest.mark.parametrize("line", [
    "Good evening, sir; your first meeting is this afternoon.",
    "Good evening, sir. I hope your afternoon meeting goes well.",
    "Good evening, sir; the evening train is cancelled.",
])
def test_a_later_hour_in_the_sentence_is_not_a_claim_about_this_one(line):
    """Only a well-wish that ENDS on the word is grounded. A fact about
    another part of the day is left exactly as the model wrote it."""
    out = ground_greeting(line, now=AT_1159)
    assert out == line.replace("Good evening", "Good morning", 1)


def test_a_tail_is_only_touched_when_its_opening_was_corrected():
    """"Good afternoon ... have a pleasant evening" at 2 pm is a wish for
    later, not a contradiction, and it stands."""
    line = "Good afternoon, sir. I hope you have a pleasant evening."
    assert ground_greeting(line, now=AT_1429) == line


# ------------------------------------------------------- somebody's words
@pytest.mark.parametrize("line", [
    "Your message reads as follows. Good morning, the meeting moved to "
    "three, sir.",
    "He wrote back. Good evening; the shipment is delayed, sir.",
    "Her note said this. Good morning, I'll be in at ten, sir.",
])
def test_a_greeting_the_reply_is_quoting_is_not_rewritten(line):
    """ground_greeting reaches summarize() and local_line() too
    (_finish_spoken with an empty user_text), which is how a mail digest
    is read out -- and a quoted "Good morning" is somebody else's hour,
    not a claim Jarvis is making."""
    assert ground_greeting(line, now=AT_1429) == line


def test_an_ordinary_second_sentence_is_still_grounded():
    """The reporting cue has to be a cue, not merely a previous sentence."""
    assert ground_greeting("It is overcast today. Good morning, sir.",
                           now=AT_1429) == \
        "It is overcast today. Good afternoon, sir."
