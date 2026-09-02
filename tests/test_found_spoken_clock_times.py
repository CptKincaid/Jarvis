"""Regression tests for clock times reaching the voice unspoken.

DEFECT (jarvis/pronounce.py):

pronounce.apply() is a vocabulary substitution map — abbreviations, jargon
and symbols — with no notion of a clock time. An ``h:mm`` token therefore
reaches the TTS engine verbatim, and XTTS reads the colon form digit by
digit. Heard in the app on 2026-08-28, answering "What's on my calendar for
Monday?": the 6:00 pm class was spoken as

    "six zero pm"

instead of "six pm" or "six o'clock". The calendar tool emits times in this
exact shape ("9:10 am BIOSENSORS at ...", "6:00 pm"), so every calendar,
alarm, reminder and briefing reply carries the defect.

Verified before the fix, straight through the real entry point:

    pronounce.apply("6:00 pm")   -> '6:00 pm'    (unchanged)
    pronounce.apply("9:10 am")   -> '9:10 am'    (unchanged)

FOLLOW-UP (same day): the first fix emitted a bare "am"/"pm", and XTTS spelled
that out as "A M". The marker is now written the way it is SAID -- "ay em" and
"pee em" -- so the engine voices it instead of spelling it.

This is a VOICE-only rewrite. jarvis/tts.py applies it in _pronounce() at
speak time, so the screen keeps "6:00 pm" and only the speaker hears words —
the same split as "the screen may be verbose, the voice may not".
"""
import pytest

from jarvis import pronounce


@pytest.mark.parametrize("written,spoken", [
    # the reported case: a whole hour drops the minutes entirely
    ("6:00 pm", "six pee em"),
    ("10:00 am", "ten ay em"),
    ("12:00 pm", "twelve pee em"),
    # minutes past the hour are read as a number
    ("9:10 am", "nine ten ay em"),
    ("12:40 pm", "twelve forty pee em"),
    ("4:10 pm", "four ten pee em"),
    # under ten past, English says "oh five", not "five"
    ("2:05 pm", "two oh five pee em"),
    ("7:01 am", "seven oh one ay em"),
    # midnight hour reads as twelve
    ("12:15 am", "twelve fifteen ay em"),
])
def test_a_clock_time_is_spoken_as_words(written, spoken):
    assert pronounce.apply(written) == spoken


def test_the_calendar_sentence_that_was_misread():
    said = pronounce.apply(
        "Magnetic Resonance Engineering again at 6:00 pm in Zachry 330.")
    assert "6:00" not in said
    assert "six pee em" in said
    # the room number is not a time and must survive untouched
    assert "330" in said


def test_am_pm_spelling_variants_are_all_caught():
    for suffix in ("pm", "PM", "p.m.", "Pm"):
        said = pronounce.apply(f"6:00 {suffix}")
        assert said.startswith("six pee em"), (suffix, said)


# 2026-09-02: this test used to assert that a bare "3:15" and a bare
# "Wisenbaker 049" were left alone, on the reasoning that without an am/pm
# marker a colon number is more likely a ratio or a score. That reasoning
# cost him a rushed line -- "Your 9:10 is Biosensors, Wisenbaker 049" at
# 08:55 -- because F5 allocates duration BY THE BYTE, so digits are
# systematically under-timed and the model compresses to fit. Both are now
# rewritten; the module comment carries the arithmetic and the evidence (18
# of 18 bare "h:mm" strings spoken across two boots were clock readings).
# What is still hands-off is what the narrowed shapes exclude.
def test_things_that_merely_look_like_times_are_left_alone():
    for text in (
            "a 16:9 aspect ratio",    # 1-digit minute: not a clock shape
            "a 4:3 crop",
            "08:56:15",               # a timestamp is not a reading
            "ETB 1035",               # no leading zero: not a room, to us
            "78 days",                # a quantity, and it must stay one
            "10 minutes",
    ):
        assert pronounce.apply(text) == text


def test_the_numbers_that_rushed_him_are_now_words():
    """2026-09-02 08:55:50, the exact string F5 was handed."""
    said = pronounce.apply(
        "While you were out, sir: one message. "
        "Your 9:10 is Biosensors, Wisenbaker 049.")
    assert said == ("While you were out, sir: one message. "
                    "Your nine ten is Biosensors, "
                    "Wisenbaker zero four nine.")


def test_it_still_does_its_original_vocabulary_job():
    assert pronounce.apply("ENGR") != "ENGR"


# 2026-08-28, later the same day: the rewrites above are XTTS COMPENSATIONS,
# not universal improvements. Fish's s2.1-pro normalises "9:10 am" correctly
# on its own, and voices the "ay em" spelling as "I'm" — the user heard it.
# Verified by listening to five variants through Fish: raw digits, dotted
# a.m., and words+capitals all read correctly; only the rewrite was wrong.
def test_the_rewrites_can_be_switched_off_per_engine():
    assert pronounce.apply("6:00 pm", rewrite_times=False) == "6:00 pm"
    assert pronounce.apply("BIOSENSORS", unshout_words=False) == "BIOSENSORS"


def test_a_strong_engine_gets_the_raw_text():
    # The per-engine table moved from jarvis/tts.py to pronounce.ENGINE_RULES
    # on 2026-09-02, where one boolean became four -- Breeze-TTS-2 reads a
    # colon fine but has no duration floor, which the old flag could not say.
    # The claim this test has always made is unchanged.
    assert pronounce.rules_for("fish").marked_times is False
    assert pronounce.rules_for("xtts").marked_times is True


def test_the_engine_decides_what_pronounce_does(tmp_path):
    from jarvis.tts import TTS
    fish = TTS(engine="fish", cache_dir=tmp_path / "a")
    xtts = TTS(engine="xtts", cache_dir=tmp_path / "b")
    line = "Your meeting is at 9:10 am."
    assert "ay em" not in fish._pronounce(line), "fish must get raw text"
    assert "nine ten ay em" in xtts._pronounce(line), "xtts still needs it"
