"""Tests for jarvis.commander.IntentClassifier — YES/NO/UNCERTAIN + learning."""
import json

import pytest

from jarvis.commander import IntentClassifier


@pytest.fixture
def clf(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    return IntentClassifier()


# ------------------------------------------------------------------ YES
def test_short_command_verbs_are_yes(clf):
    for text in ("run tests", "fix it", "commit now", "take a screenshot"):
        label, conf = clf.classify(text)
        assert label == IntentClassifier.YES, text
        assert conf >= 0.8


def test_question_with_positive_patterns_is_yes(clf):
    label, conf = clf.classify("can you fix the bug in the parser?")
    assert label == IntentClassifier.YES
    assert conf >= 0.7


def test_short_question_mark_is_yes(clf):
    label, conf = clf.classify("why though?")
    assert label == IntentClassifier.YES
    assert conf == 0.8


# ------------------------------------------------------------------- NO
def test_empty_and_tiny_text_is_no(clf):
    assert clf.classify("") == (IntentClassifier.NO, 1.0)
    assert clf.classify("hi") == (IntentClassifier.NO, 1.0)


def test_short_reaction_is_no(clf):
    label, conf = clf.classify("yeah totally")
    assert label == IntentClassifier.NO
    assert conf == 0.8


def test_casual_chatter_is_no(clf):
    label, _ = clf.classify("oh my god that's crazy lol haha")
    assert label == IntentClassifier.NO


def test_third_person_gossip_is_no(clf):
    label, _ = clf.classify("she said he told them about her party yesterday")
    assert label == IntentClassifier.NO


# ------------------------------------------------------------ UNCERTAIN
def test_no_signals_is_uncertain(clf):
    label, conf = clf.classify("banana purple elephant dancing")
    assert label == IntentClassifier.UNCERTAIN
    assert conf == 0.5


# ----------------------------------------------------- feedback learning
def test_feedback_learns_positive(clf):
    text = "banana purple elephant dancing"
    assert clf.classify(text)[0] == IntentClassifier.UNCERTAIN
    clf.log_feedback(text, True)
    label, conf = clf.classify(text)
    assert label == IntentClassifier.YES
    assert conf >= 0.7


def test_feedback_learns_negative(clf):
    text = "kumquat zebra painting store today"
    clf.log_feedback(text, False)
    label, _ = clf.classify(text)
    assert label == IntentClassifier.NO


def test_feedback_persists_to_disk_and_reloads(clf):
    text = "banana purple elephant dancing"
    clf.log_feedback(text, True)
    assert clf.INTENT_LOG.exists()
    data = json.loads(clf.INTENT_LOG.read_text())
    assert data == [{"text": text, "label": "yes"}]

    fresh = IntentClassifier()          # same patched INTENT_LOG path
    assert fresh.num_examples == 1
    assert fresh.classify(text)[0] == IntentClassifier.YES


def test_feedback_flips_learned_label(clf):
    text = "banana purple elephant dancing"
    clf.log_feedback(text, True)
    assert clf.classify(text)[0] == IntentClassifier.YES
    clf.log_feedback(text, False)
    assert clf.classify(text)[0] == IntentClassifier.NO


def test_log_capped_at_500(clf):
    for i in range(510):
        clf.log_feedback(f"sample utterance number {i} words", True)
    assert clf.num_examples == 500


# ---------------------------------------------- assistant-era vocabulary
#
# _POSITIVE_PATTERNS was written when Jarvis was a coding assistant: fix,
# commit, push, refactor, "the bug", "the file". It never learned the
# vocabulary that arrived with timers, alarms, media, calendar and mail, so
# on a clean install real commands were being classified NO and discarded
# SILENTLY -- no card, no prompt, no way for the feedback loop to correct it,
# because only UNCERTAIN ever asks. Measured before the fix: "play some jazz",
# "volume up", "next track" and "snooze" were all NO 0.80, and "remind me to
# call the supplier at four" was NO 1.00.
#
# A false NO is the expensive error here. Anything reaching this classifier
# already cleared the wake word AND the speaker gate, so it is the enrolled
# user talking to Jarvis on purpose; refusing to hear them is worse than
# occasionally acting on something they were only half-asking for.
ASSISTANT_COMMANDS = [
    "set a timer for ten minutes", "set an alarm for seven am",
    "remind me to call the supplier at four", "wake me at six thirty",
    "snooze", "cancel the timer",
    "play some jazz", "play some miles davis", "pause the music",
    "next track", "skip this song", "volume up", "turn the volume down",
    "mute", "what time is it", "what's the weather tomorrow",
    "what's on my calendar today", "read me the last message",
    "summarize my day for me", "add a note about the supplier",
    "check my email", "how long until the meeting",
]

BACKGROUND_CHATTER = [
    "oh my god that's crazy lol haha",
    "she said he told them about her party yesterday",
    "yeah totally", "hi", "",
]


@pytest.mark.parametrize("text", ASSISTANT_COMMANDS)
def test_assistant_commands_are_not_discarded(clf, text):
    """The regression that mattered: never silently NO on a real command."""
    label, conf = clf.classify(text)
    assert label != IntentClassifier.NO, f"{text!r} would be silently dropped"


@pytest.mark.parametrize("text", ASSISTANT_COMMANDS)
def test_assistant_commands_route_without_asking(clf, text):
    """...and they should not need a card either; that is just friction."""
    label, conf = clf.classify(text)
    assert label == IntentClassifier.YES, f"{text!r} -> {label} {conf}"


@pytest.mark.parametrize("text", BACKGROUND_CHATTER)
def test_background_chatter_is_still_dropped(clf, text):
    assert clf.classify(text)[0] == IntentClassifier.NO, text


def test_length_alone_is_not_evidence_against_a_command(clf):
    """`len(words) >= 8 and pos_score == 0` used to push long unmatched
    commands to NO. Length is not evidence about who you are talking to."""
    label, _ = clf.classify("remind me to call the supplier at four o'clock")
    assert label == IntentClassifier.YES
