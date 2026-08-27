"""Reading a spoken yes/no after "Was that for me?".

When intent is uncertain, Jarvis now asks aloud and opens a short listening
window. Whatever comes back is read by `parse_yes_no` ONLY -- it is never
routed as a command. That restriction is the whole safety property: if an
ambiguous reply were dispatched normally it could classify as uncertain
again, and the two prompts would ping-pong.

So the parser must return None for anything it does not clearly recognise,
and the caller leaves the card on screen for a click. Being wrong in the
"I didn't catch that" direction costs a click; being wrong in the other
direction runs a command the user did not ask for.
"""
from __future__ import annotations

import pytest

from jarvis.commander import parse_yes_no


@pytest.mark.parametrize("text", [
    "yes", "Yes.", "  YES  ", "yeah", "yep", "yup", "sure",
    "correct", "affirmative", "go ahead", "please do",
    "yes it was", "yeah that was for you",
])
def test_affirmatives(text):
    assert parse_yes_no(text) is True


@pytest.mark.parametrize("text", [
    "no", "No!", "nope", "nah", "negative",
    "never mind", "nevermind", "ignore that", "forget it",
    "no it wasn't", "not for you",
])
def test_negatives(text):
    assert parse_yes_no(text) is False


@pytest.mark.parametrize("text", [
    "", "   ", None,
    "what's the weather tomorrow",
    "open the terminal",
    "she said no way lol haha dude",     # the repo's own ambiguity fixture
])
def test_not_an_answer_returns_none(text):
    assert parse_yes_no(text) is None


@pytest.mark.parametrize("text", [
    "nothing much", "north of here", "you know", "notice that",
    "snowball", "another one",
])
def test_no_is_not_matched_inside_other_words(text):
    """Substring matching would make 'nothing' and 'you know' mean no."""
    assert parse_yes_no(text) is not False


@pytest.mark.parametrize("text", [
    "not sure", "unsure", "i don't know", "dunno", "maybe",
])
def test_explicit_uncertainty_is_not_an_answer(text):
    """'not sure' contains 'sure'; it must not read as yes."""
    assert parse_yes_no(text) is None


def test_contradictory_reply_is_not_an_answer():
    assert parse_yes_no("yes and no") is None
    assert parse_yes_no("no, yes, wait") is None


def test_answer_can_be_embedded_in_a_sentence():
    assert parse_yes_no("yes, go ahead and do that") is True
    assert parse_yes_no("no, I was talking to the dog") is False
