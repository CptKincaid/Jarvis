"""Tier-1 spoken arithmetic and unit conversion (jarvis/mathspeak.py) and
its wiring into the commander's ladder.

Pure and offline: no clock, no services, no model. The interesting tests
are property-style -- a table of phrasings is only ever as good as the
phrasings someone thought of, so conversions are checked by round trip and
against physical constants, and the "never claim what you cannot answer"
rule is checked against a corpus of real non-maths utterances taken from
the rest of the suite's Tier-1 samples.
"""
from __future__ import annotations

import random

import pytest

from jarvis import mathspeak as ms
from jarvis.commander import Commander, math_kind
from tests.test_commander import rich, services  # noqa: F401  (fixtures)


# ------------------------------------------------------------ numbers
@pytest.mark.parametrize("text,value", [
    ("74", 74.0), ("13.5", 13.5), ("1,250", 1250.0),
    ("eighteen", 18.0), ("seventy four", 74.0), ("seventy-four", 74.0),
    ("a hundred and twenty", 120.0), ("one hundred twenty", 120.0),
    ("three point five", 3.5), ("two thousand", 2000.0),
    ("a", 1.0), ("an", 1.0), ("one and a half", 1.5),
    ("twelve million", 12_000_000.0),
    ("", None), ("banana", None), ("the exam", None), ("and", None),
])
def test_parse_number(text, value):
    assert ms.parse_number(text) == value


def test_number_words_agree_with_digits_for_every_small_integer():
    """Property: the word form and the digit form must parse identically."""
    words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
             11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
             15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
             19: "nineteen"}
    tens = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
            70: "seventy", 80: "eighty", 90: "ninety"}
    for n in range(0, 20):
        assert ms.parse_number(words[n]) == float(n)
    for t, tw in tens.items():
        for n in range(0, 10):
            spoken = tw if n == 0 else f"{tw} {words[n]}"
            assert ms.parse_number(spoken) == float(t + n), spoken


# --------------------------------------------------------- conversion
def test_every_unit_round_trips_within_its_dimension():
    """Property: converting there and back is the identity, for every pair
    of units that share a dimension -- temperature's offsets included."""
    rng = random.Random(20260830)
    for a in ms.UNITS.values():
        for b in ms.UNITS.values():
            if a.dim != b.dim:
                assert ms.convert(1.0, a, b) is None
                continue
            for _ in range(3):
                v = rng.uniform(0.5, 5000.0)
                there = ms.convert(v, a, b)
                back = ms.convert(there, b, a)
                assert back == pytest.approx(v, rel=1e-9), (a.key, b.key, v)


@pytest.mark.parametrize("value,src,dst,expected", [
    (1.0, "kg", "lb", 2.2046226),        # the definition of the pound
    (1.0, "mi", "km", 1.609344),         # exact by international agreement
    (1.0, "ft", "in", 12.0),
    (1.0, "st", "lb", 14.0),
    (0.0, "c", "f", 32.0),
    (100.0, "c", "f", 212.0),
    (-40.0, "c", "f", -40.0),            # the one place the scales meet
    (0.0, "c", "k", 273.15),
    (1.0, "gb", "mb", 1000.0),           # SI, as the standard says
    (1.0, "gib", "mib", 1024.0),         # binary, as the standard says
])
def test_conversions_against_the_defining_constants(value, src, dst, expected):
    got = ms.convert(value, ms.UNITS[src], ms.UNITS[dst])
    assert got == pytest.approx(expected, rel=1e-6)


# -------------------------------------------------------------- solve
@pytest.mark.parametrize("text,expected", [
    ("what's 18 percent of 74", "18 percent of 74 is 13.32, sir."),
    ("what is 18% of 74", "18 percent of 74 is 13.32, sir."),
    ("what's eighteen percent of seventy four", "18 percent of 74 is 13.32, sir."),
    ("what's 15 percent off 80", "15 percent off 80 is 68, sir."),
    ("what's 43 times 17", "43 times 17 is 731, sir."),
    ("calculate 43 x 17", "43 times 17 is 731, sir."),
    ("what's 20 plus 22", "20 plus 22 is 42, sir."),
    ("what's seventy four minus eighteen", "74 minus 18 is 56, sir."),
    ("what's 7 divided by 2", "7 divided by 2 is 3.5, sir."),
    ("what's the square root of 144", "The square root of 144 is 12, sir."),
    ("how many ounces in 300 grams", "300 grams is 10.58 ounces, sir."),
    ("convert 5 miles to kilometres", "5 miles is 8.05 kilometres, sir."),
    ("300 grams in ounces", "300 grams is 10.58 ounces, sir."),
    ("what's 100 fahrenheit in celsius",
     "100 degrees Fahrenheit is 37.78 degrees Celsius, sir."),
    ("how many megabytes in a gigabyte", "1 gigabyte is 1000 megabytes, sir."),
    ("jarvis, what is 2 kg in pounds?", "2 kilograms is 4.41 pounds, sir."),
])
def test_spoken_answers(text, expected):
    ans = ms.solve(text)
    assert ans is not None and ans.ok and ans.text == expected


@pytest.mark.parametrize("text,fragment", [
    ("how many ounces in five miles", "don't convert"),
    ("what's 10 divided by 0", "divide by zero"),
    ("convert 20 dollars to euros", "currency"),
    ("how many dollars in 20 euros", "currency"),
    ("what's 5 pounds in dollars", "currency"),
    ("what's the square root of minus nine", "square root of a negative"),
])
def test_honest_refusals_are_spoken_not_guessed(text, fragment):
    ans = ms.solve(text)
    assert ans is not None and not ans.ok and fragment in ans.text


# A corpus of things people actually say to Jarvis that must NEVER be
# claimed here: every one of them has a real owner further down the ladder.
NOT_MATHS = [
    "set a timer for five minutes",
    "remind me to call mum at five",
    "how many reminders do i have",
    "how many emails did i get",
    "how many gigabytes of memory do i have",
    "how much memory is free",
    "how much disk space is left",
    "what's the time in london",
    "what's the weather",
    "convert this file to markdown",
    "how many days until my exam",
    "what can you do",
    "how many blocks have i done",
    "how many dollars did i spend",
    "quiz me on chapter three",
    "read the clipboard",
    "what's on my todo list",
    "how long left",
    "give me my briefing",
    "add buy milk to my todo list",
    "run diagnostics",
    "what did i say about the thesis",
    "how's my week looking",
    "start a focus session",
    "cancel the timer",
    "what's your ip address",
    "how did yesterday go",
    "two plus three times four",          # a chain: precedence is ambiguous
]


@pytest.mark.parametrize("text", NOT_MATHS)
def test_never_claims_an_utterance_it_cannot_answer(text):
    assert ms.solve(text) is None, text


def test_a_broken_parse_never_breaks_the_ladder(monkeypatch):
    monkeypatch.setattr(ms, "_solve_convert",
                        lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ms.solve("what's 18 percent of 74") is None


# ------------------------------------------------------------ wording
@pytest.mark.parametrize("value,text", [
    (731.0, "731"), (13.32, "13.32"), (3.5, "3.5"), (0.0, "0"),
    (12.0, "12"), (1250.0, "1250"), (0.001234, "0.0012"),
    (-40.0, "-40"), (2.204622, "2.2"),
])
def test_fmt_number_reads_aloud(value, text):
    assert ms.fmt_number(value) == text


def test_no_thousands_commas_reach_the_voice():
    """Property: the TTS reads a comma as a pause, so a number must never
    carry one -- but the parser must still accept one from a transcript."""
    assert "," not in ms.fmt_number(1234567.0)
    ans = ms.solve("what's 1,250 times 2")
    assert ans is not None and ans.text == "1250 times 2 is 2500, sir."


def test_singular_units_are_spoken_singular():
    ans = ms.solve("how many inches in 2.54 centimetres")
    assert ans is not None and ans.text.endswith("is 1 inch, sir.")


# ---------------------------------------------------------- the ladder
def test_math_kind_is_the_evaluator_itself():
    """The matcher returns the finished answer, so the handler can never be
    handed a half-parse it has to guess about."""
    assert math_kind("what's 43 times 17").text == "43 times 17 is 731, sir."
    assert math_kind("set a timer for five minutes") is None


def test_the_commander_answers_a_sum_without_the_wake_word():
    c = object.__new__(Commander)
    assert c._match_assistant("what's 18 percent of 74") == "math"
    assert c._match_assistant("how many ounces in 300 grams") == "math"
    assert c._match_assistant("set a timer for five minutes") != "math"


def test_a_sum_is_spared_the_intent_classifier(rich, monkeypatch):  # noqa: F811
    """A Tier-1 match must never be handed to the gate -- a false NO there
    is silent and unrecoverable."""
    def boom(*a, **k):
        raise AssertionError("the classifier was consulted for a sum")
    monkeypatch.setattr(rich.intent, "classify", boom)
    res = rich.handle("what's 43 times 17", source="voice")
    assert res.handled and res.speak and res.reply == "43 times 17 is 731, sir."


def test_the_refusal_lines_are_fixed_strings_the_app_can_prewarm():
    for line in ms.PERSONA_LINES:
        assert isinstance(line, str) and line and "{" not in line
