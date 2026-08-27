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
