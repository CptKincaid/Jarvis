"""#8 "doesnt understand say again. or any of these".

From the 2026-08-31 voice session (/tmp/vss_voice/jarvis.log), verbatim:

    20:36:36.775 jarvis.commander INFO handle 'Say again' source=voice
    20:36:36.775 jarvis.commander INFO Ignored (background chat, conf=0.80): 'Say again'
    20:37:01.649 jarvis.commander INFO handle 'Say again' source=voice
    20:37:01.650 jarvis.commander INFO Ignored (background chat, conf=0.80): 'Say again'

Twice, thirty seconds apart, the words reached the commander CORRECTLY
transcribed and were thrown away without a sound.

Why: the hotword consumes the wake word, so a spoken "Jarvis, say again"
arrives here as the bare "Say again" — `strip_jarvis_prefix` returns None,
the prefixed registry pass (which owns Command("repeat", ...)) is skipped,
and the utterance falls to the intent classifier, which calls every one- or
two-word phrase NO. The Tier-1 voice-I/O block that DOES answer these lives
in `_route_text`, one rung BELOW the gate, so it was unreachable by voice.

`_match_assistant` could not save it either: ASSISTANT_TIER1 is a
name-filtered view of REGISTRY and neither "repeat" nor "quiet" is in it.

"quiet"/"stop" is the SAME hole and already has its own rung (3'', added
for his #9); the fix here is its twin, placed beside it. The quiet cases
below are a standing guard, not a new fix — they are here because "say
again" and "stop" are the two words a barge-in is made of and the pair must
never drift apart again.

Two of the phrases he listed, "say that again" and "what was that", DID
survive the night — by accidentally matching `read_control_kind`, which is
in ASSISTANT_TIER1. That is luck, not design: it does not cover "say
again", "repeat that", "come again" or "pardon", which is why his note says
"or any of these".
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from jarvis.commander import ASSISTANT_TIER1, Commander, IntentClassifier
from jarvis.config import CONFIG


@pytest.fixture
def cmdr(tmp_path, monkeypatch):
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(),
    )
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    # The last thing Jarvis said, which "say again" replays.
    svc.tts.last_text = "The time is ten past eight, sir."
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    return Commander(svc)


def _always_background(monkeypatch):
    """The classifier as it actually behaved on the night: a confident NO
    on the short phrases. Pinned so the test cannot pass because the live
    ~/.aiws_trainer/intent_log.json happened to have learned one of them."""
    monkeypatch.setattr(IntentClassifier, "classify",
                        lambda self, text: (IntentClassifier.NO, 0.80))


# ------------------------------------------------------------------ #8
@pytest.mark.parametrize("said", [
    "Say again",            # the transcript in the log, twice
    "Say again?",
    "say that again",       # his note: "or any of these"
    "repeat that",
    "come again",
    "what was that",
])
def test_say_again_is_not_background_chat(cmdr, monkeypatch, said):
    _always_background(monkeypatch)
    res = cmdr.handle(said, source="voice")
    assert res.status != "Ignored (background chat)", \
        f"{said!r} was dropped in silence again"
    assert res.status == "Repeating"
    assert res.reply == "The time is ten past eight, sir."


def test_say_again_with_nothing_said_yet_still_answers(cmdr, monkeypatch):
    """Even the empty case must be a sentence, not silence."""
    _always_background(monkeypatch)
    cmdr.services.tts.last_text = ""
    res = cmdr.handle("say again", source="voice")
    assert res.status == "Nothing to repeat"


# ------------------------------------------------- its twin, rung 3'' (#9)
@pytest.mark.parametrize("said", ["stop", "quiet", "be quiet", "that's enough"])
def test_the_word_a_barge_in_ends_with_reaches_the_handler(cmdr, monkeypatch,
                                                           said):
    """Already fixed (#9); pinned so the pair cannot drift apart."""
    _always_background(monkeypatch)
    res = cmdr.handle(said, source="voice")
    assert res.status != "Ignored (background chat)"
    assert res.status == "Quiet"
    # Shown, never spoken: he has just told Jarvis to be quiet.
    assert res.speak is False
    cmdr.services.tts.interrupt.assert_called()


# ------------------------------------------------------- the guard itself
def test_voice_io_is_still_absent_from_assistant_tier1():
    """Why both rungs are spelled out rather than left to _match_assistant.

    ASSISTANT_TIER1 is a name-filtered view of REGISTRY, and neither entry
    is in the list — which is what made both phrases invisible to the
    intent gate's Tier-1 bypass. Asserted rather than assumed: if a later
    change adds them there, _try_assistant would start running them ahead
    of the read-aloud transport words ("stop" while reading), and the rungs
    above would silently stop being the thing that answers them."""
    names = {c.name for c in ASSISTANT_TIER1}
    assert "repeat" not in names and "quiet" not in names
