"""FOUND 2026-08-26 (routing sweep): "tool" / "tools" is missing from
jarvis.router.CODE_OBJECTS, so a coding request whose only object noun is
"tool" collapses to the weak local topic beside it and routes LOCAL.

This is the spec's own worked example of an ambiguous utterance: a CODE
task that happens to contain a local topic noun.  Rule trace for
"check the weather tool":

    _VERB_RX   -> ['check']          (a coding verb)
    _OBJECT_RX -> []                 ("tool" is not a CODE_OBJECT)
    _PATH_RX   -> []
    => code_cues() = (strong=False, weak=False)
    _LOCAL_WEAK_RX -> 'weather'
    => Cues(code_strong=False, code_weak=False, local_weak=True)
    => route(): "local_weak and not code_weak" -> RouteDecision("local",
       "local:topic")

Measured with the real Router built on the real AssistantConfig, no
classifier stub reached (the rules decide before rule 6):

    check the weather tool        -> local  (local:topic)
    fix the weather tool          -> local  (local:topic)
    fix the alarm tool            -> local  (local:topic)
    rewrite the notes tool        -> local  (local:topic)
    debug the spotify tool        -> local  (local:music)

The same sentences with a noun that IS in CODE_OBJECTS route correctly,
which isolates the cause to the missing word:

    fix the weather module        -> claude (code-cue)
    fix the weather handler       -> claude (code-cue)
    fix the mail parser           -> claude (code-cue)

Defect: jarvis/router.py CODE_OBJECTS (the tuple beginning "code",
"codebase", ...) has no "tool"/"tools" entry.

FIXED 2026-08-26: "tool"/"tools" added to CODE_OBJECTS.  "debug the
spotify tool" needed one thing more -- a bare service name that MODIFIES a
code object ("the spotify tool") is a piece of software, not an errand, so
local_cues() now demotes such a match to a weak topic noun (_CODE_HEAD_RX)
and the coding pair beside it wins.  This test is no longer xfail.
"""
import pytest

from jarvis.router import Router

CODE_TASKS_WITH_A_LOCAL_NOUN = [
    "check the weather tool",
    "fix the weather tool",
    "fix the alarm tool",
    "rewrite the notes tool",
    "debug the spotify tool",
]


@pytest.mark.parametrize("text", CODE_TASKS_WITH_A_LOCAL_NOUN)
def test_a_code_task_about_a_tool_goes_to_claude(text):
    d = Router(None, classify=None).route(text)
    assert d.kind == "claude", (text, d.kind, d.reason)


def test_the_same_sentence_with_a_known_code_object_routes_correctly():
    """Control: isolates the cause to the missing CODE_OBJECTS entry."""
    for text in ("fix the weather module", "fix the weather handler",
                 "fix the mail parser"):
        d = Router(None, classify=None).route(text)
        assert d.kind == "claude", (text, d.kind, d.reason)
