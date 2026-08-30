"""Voice feedback (2026-08-30): "that was for you" / "that wasn't for you".

The intent classifier's only learning signal used to be the YES/NO card on
UNCERTAIN utterances; a command dropped as background chat (NO) was silent
and unrecoverable. These phrases label the LAST turn in the classifier's
own log, re-run a dropped command, and stop a reply he did not ask for.
They are matched ahead of the classifier -- three-word feedback would
itself be classified NO.

Real Router and IntentClassifier (log at tmp_path); mocked services.
"""
import json
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.commander import Commander, IntentClassifier, feedback_kind, parse_yes_no
from jarvis.config import CONFIG
from jarvis.router import Router


class Cfg:
    def __init__(self, **over):
        self.data = {"claude.big_model": "fable", "briefing.enabled": False}
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


@pytest.fixture
def rich(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    # raising=False: FEEDBACK_LOG arrives with the voice-feedback feature,
    # and this file must run on the commits before it too.
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "feedback.jsonl",
                        raising=False)
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(), assistant=Cfg(),
        timekeeper=MagicMock(), notes=MagicMock(), approvals=MagicMock(),
        claude=MagicMock(),
        conversation=SimpleNamespace(forget_exchange=MagicMock(return_value=True)))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    return Commander(svc), svc


def _intent_log(tmp_path):
    return json.loads((tmp_path / "intent_log.json").read_text())


def _feedback_lines(tmp_path):
    path = tmp_path / "feedback.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


DROPPED = "what's the weather like tomorrow"


def _drop(c, monkeypatch, text=DROPPED):
    """A voice utterance the classifier calls background chat."""
    monkeypatch.setattr(c.intent, "classify", lambda t: (IntentClassifier.NO, 0.8))
    res = c.handle(text, source="voice")
    assert res.status == "Ignored (background chat)"
    return res


def test_that_was_for_you_relabels_and_reruns_a_dropped_command(rich, tmp_path, monkeypatch):
    c, svc = rich
    _drop(c, monkeypatch)
    svc.brain.chat.assert_not_called()
    seen = []
    monkeypatch.setattr(c.intent, "classify",
                        lambda t: seen.append(t) or (IntentClassifier.NO, 0.8))

    res = c.handle("that was for you", source="voice")

    assert seen == [], "feedback and the re-run must not go through the classifier"
    assert res.corrected == DROPPED
    assert svc.brain.chat.call_args.args == (DROPPED,)
    assert _intent_log(tmp_path) == [{"text": DROPPED, "label": "yes"}]
    line, = _feedback_lines(tmp_path)
    assert line["text"] == DROPPED and line["label"] == "yes"
    assert line["prev_status"].startswith("Ignored")


def test_the_prefixed_form_works_too(rich, monkeypatch):
    c, svc = rich
    _drop(c, monkeypatch)
    res = c.handle("Jarvis, that was for you.", source="voice")
    assert res.corrected == DROPPED
    assert svc.brain.chat.call_args.args == (DROPPED,)


def test_the_window_after_a_dropped_command_is_short(rich, monkeypatch, tmp_path):
    """No reply marks the time of a dropped command, so "that was for you"
    a minute later has nothing sure to point at."""
    c, svc = rich
    _drop(c, monkeypatch)
    c._last_turn.ts -= 30
    res = c.handle("that was for you", source="voice")
    assert res.status == "Feedback: stale" and res.speak
    svc.brain.chat.assert_not_called()
    assert not (tmp_path / "intent_log.json").exists()


def test_that_wasnt_for_you_after_a_reply_labels_no_and_stops_him(rich, tmp_path):
    c, svc = rich
    c.handle(DROPPED, source="typed")            # answered (a brain turn)
    res = c.handle("that wasn't for you", source="typed")
    assert res.reply == "My mistake, sir." and res.speak
    assert _intent_log(tmp_path) == [{"text": DROPPED, "label": "no"}]
    svc.tts.interrupt.assert_called_once()
    svc.conversation.forget_exchange.assert_called_once_with(DROPPED)


def test_that_wasnt_for_you_after_a_drop_confirms_the_no(rich, monkeypatch, tmp_path):
    c, svc = rich
    _drop(c, monkeypatch)
    res = c.handle("no, that wasn't for you", source="voice")
    assert res.reply == "Very good, sir."
    assert _intent_log(tmp_path) == [{"text": DROPPED, "label": "no"}]
    svc.brain.chat.assert_not_called()


def test_that_was_for_you_after_a_reply_confirms_the_yes_without_rerunning(rich, tmp_path):
    c, svc = rich
    c.handle(DROPPED, source="typed")
    svc.brain.chat.reset_mock()
    res = c.handle("that was for you", source="typed")
    assert res.reply == "Very good, sir." and res.corrected is None
    svc.brain.chat.assert_not_called()
    assert _intent_log(tmp_path) == [{"text": DROPPED, "label": "yes"}]


def test_after_was_that_for_me_the_card_is_claimed_and_the_text_runs(rich, monkeypatch, tmp_path):
    c, svc = rich
    monkeypatch.setattr(c.intent, "classify", lambda t: (IntentClassifier.UNCERTAIN, 0.5))
    c.on_uncertain = lambda text: None
    c.claim_uncertain = MagicMock(return_value=True)
    asked = c.handle(DROPPED, source="voice")
    assert asked.status == "Was that for me?"

    res = c.handle("that was for you", source="voice")

    c.claim_uncertain.assert_called_once_with(True)
    assert res.corrected == DROPPED
    assert svc.brain.chat.call_args.args == (DROPPED,)
    assert _intent_log(tmp_path) == [{"text": DROPPED, "label": "yes"}]


def test_feedback_with_nothing_to_point_at(rich):
    c, svc = rich
    res = c.handle("that was for you", source="typed")
    assert res.status == "Feedback: no last turn" and res.speak
    svc.brain.chat.assert_not_called()


def test_the_learned_label_changes_the_next_classification(rich, monkeypatch):
    """The point of the exercise: the classifier's own log is fed, so the
    same words are not dropped again."""
    c, svc = rich
    _drop(c, monkeypatch, "drop the needle on some jazz")
    monkeypatch.undo()
    # a fresh classifier reading the log the feedback wrote
    c.handle("that was for you", source="typed")
    fresh = IntentClassifier()
    assert "drop the needle" in fresh._learned_positive


@pytest.mark.parametrize("text,label", [
    ("that was for you", True),
    ("Jarvis, that was for you", True),
    ("yes, that was to you", True),
    ("I was talking to you", True),
    ("that was a command", True),
    ("that wasn't for you", False),
    ("no, that was not for you", False),
    ("I wasn't talking to you", False),
    ("talking to someone else", False),
    ("not you", False),
    ("that was fun", None),
    ("what's the weather", None),
])
def test_feedback_kind(text, label):
    assert feedback_kind(text) is label


def test_parse_yes_no_accepts_that_was_for_you_in_the_spoken_window():
    """_ask_uncertain listens 5 s with parse_yes_no: the natural reply to
    "Was that for me?" is "that was for you", which used to be neither."""
    assert parse_yes_no("that was for you") is True
    assert parse_yes_no("yes, that was for you") is True
    assert parse_yes_no("that wasn't for you") is False
    assert parse_yes_no("no, that was not for you") is False
