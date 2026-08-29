"""Regression tests: ALL-CAPS words read as acronyms.

DEFECT (jarvis/pronounce.py):

Calendar titles come straight from iCal, and TAMU course names are shouted --
"9:10 am BIOSENSORS at College Station Wisenbaker Engineering Bldg 049". TTS
engines treat an all-caps token as an acronym, so XTTS did not say the word
"biosensors"; it produced something the user heard as "bio censors"
(2026-08-28). Every shouted course name on the calendar has the same problem.

The vocabulary table cannot solve this by enumeration: the titles are Hunter's,
they change every semester, and there is no list to keep.

The rule instead: an all-caps run that READS as a word -- several letters, with
real vowels in it -- is a shouted word and gets title case. An all-caps run
without the vowels to carry a syllable ("HDMI", "PHYS", "ETB") is an initialism
and is left exactly as it was, so the table and the engine can still spell it.
"""
import pytest

from jarvis import pronounce


@pytest.mark.parametrize("shouted,spoken", [
    ("BIOSENSORS", "Biosensors"),
    ("THERMODYNAMICS", "Thermodynamics"),
    ("CALCULUS", "Calculus"),
    ("SEMINAR", "Seminar"),
])
def test_a_shouted_word_is_spoken_as_a_word(shouted, spoken):
    assert pronounce.apply(shouted) == spoken


@pytest.mark.parametrize("initialism", ["ETB", "PHYS", "MWF", "TR"])
def test_an_initialism_is_left_for_the_engine_to_spell(initialism):
    assert initialism in pronounce.apply(initialism)


def test_the_reported_calendar_line():
    said = pronounce.apply(
        "9:10 am BIOSENSORS at College Station Wisenbaker Engineering Bldg 049")
    assert "BIOSENSORS" not in said
    assert "Biosensors" in said
    # and the rest of the line still gets its usual treatment
    assert "nine ten ay em" in said
    assert "Building" in said


def test_known_jargon_still_wins_over_the_caps_rule():
    # these are in the table and must keep their spelled-out forms
    assert pronounce.apply("VSS") == "V S S"
    assert pronounce.apply("CUDA") == "kooda"
    assert pronounce.apply("DGX") == "D G X"


def test_ordinary_text_is_untouched():
    text = "You have four items on Monday, sir."
    assert pronounce.apply(text) == text
