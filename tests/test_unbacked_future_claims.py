"""Drop into tests/ as test_unbacked_future_claims.py after the patch."""
import pytest
from jarvis import brain

@pytest.mark.parametrize("line", [
    "I shall pass on your regards to Ali and Heather.",
    "I'll let Heather know, sir.",
    "I will send that along now.",
    "I'll add milk to the list.",
    "I shall set a timer for ten minutes.",
])
def test_a_promise_to_act_is_an_unbacked_claim(line):
    """2026-09-04 15:06:29: a promise no tool could keep, naming two real people,
    walked through a table that knew only the past and the progressive."""
    assert brain.unbacked_claim(line), line

@pytest.mark.parametrize("line", [
    "I'll be here, sir.",                 # idiom, no object
    "I'll stop there.",                   # idiom veto
    "I won't send anything without you.", # negated
    "I will say this much: it is late.",  # 'say' is speech, not an action
    "Nothing will be sent, sir.",         # passive, negated
])
def test_ordinary_future_speech_is_not_a_claim(line):
    assert brain.unbacked_claim(line) is None, line

def test_the_regards_line_is_replaced_and_the_rest_kept():
    text = ("You are in College Station, Texas, sir. I shall pass on your regards "
            "to Ali and Heather. Shall I run your briefing?")
    out = brain.strip_unbacked_claims(text)
    assert "regards" not in out and brain.UNBACKED_LINE in out
    assert "College Station" in out and "briefing" in out
