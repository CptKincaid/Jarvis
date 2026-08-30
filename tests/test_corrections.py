"""Corrections (2026-08-30): "no, I said X" / "I meant X" / "not Y, X".

The commander re-dispatches X with the intent gate bypassed, cuts the
reply in flight (TTS and the brain job), forgets the misheard exchange
from the conversation window and logs the (heard, meant) pair. The
brain's job generation keeps a cancelled job's reply from being spoken or
remembered once the corrected job has started.

Commander tests: real Router, mocked services. Brain test: the real
JarvisBrain with _chat_sync stubbed. No Ollama, no network.
"""
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.brain as brain_mod
from jarvis.commander import Commander, IntentClassifier, correction_kind, new_vocab_words
from jarvis.config import CONFIG, PATHS
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
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
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


HEARD = "what's on my calendar tomorrow"
MEANT = "what's the weather tomorrow"


def test_a_correction_re_dispatches_the_meant_text_and_cleans_up(rich):
    c, svc = rich
    c.handle(HEARD, source="typed")
    svc.brain.chat.reset_mock()

    res = c.handle(f"no, I said {MEANT}", source="typed")

    assert res.corrected == MEANT
    assert svc.brain.chat.call_count == 1
    assert svc.brain.chat.call_args.args == (MEANT,)
    # the reply in flight is cut: speech now, the model job too
    svc.tts.interrupt.assert_called_once()
    svc.brain.cancel.assert_called_once()
    # the misheard exchange leaves the model's window
    svc.conversation.forget_exchange.assert_called_once_with(HEARD)
    # and the pair is logged for a later vocab / prompt tune
    svc.memory.log_correction.assert_called_once_with(HEARD, MEANT)
    # the meant text is now the last turn: a second "no, I said" corrects IT
    assert c._last_turn.text == MEANT


def test_by_voice_the_correction_bypasses_the_intent_gate(rich, monkeypatch):
    """A follow-up utterance still runs the classifier, which calls short
    phrases background chat. The correction is addressed to Jarvis by
    construction and must not be subject to that guess."""
    c, svc = rich
    monkeypatch.setattr(c.intent, "classify", lambda text: (IntentClassifier.NO, 0.8))
    first = c.handle(HEARD, source="voice")
    assert first.status.startswith("Ignored")
    seen = []
    monkeypatch.setattr(c.intent, "classify",
                        lambda text: seen.append(text) or (IntentClassifier.NO, 0.8))

    res = c.handle(f"no, I said {MEANT}", source="voice")

    assert res.corrected == MEANT
    assert seen == [], "the classifier must not see the corrected text"
    assert svc.brain.chat.call_args.args == (MEANT,)


def test_by_voice_a_correction_needs_a_recent_turn(rich, monkeypatch):
    """Without a turn to correct, "not that one, the other one" is the room
    talking: it goes through the classifier like anything else."""
    c, svc = rich
    monkeypatch.setattr(c.intent, "classify", lambda text: (IntentClassifier.NO, 0.8))
    res = c.handle(f"no, I said {MEANT}", source="voice")
    assert res.status.startswith("Ignored")
    svc.brain.chat.assert_not_called()
    # a turn a long time ago does not count either
    c.handle(HEARD, source="typed")
    c._last_turn.ts -= 600
    svc.brain.chat.reset_mock()
    res = c.handle(f"no, I said {MEANT}", source="voice")
    assert res.status.startswith("Ignored")
    svc.brain.chat.assert_not_called()


def test_a_correction_is_not_eaten_as_a_decline(rich):
    """parse_yes_no reads any sentence opening with "no" as a no, so with a
    permission question pending "no, I said X" used to decline it. The
    correction runs first; X then meets the pending question on its own."""
    c, svc = rich
    c.handle(HEARD, source="typed")
    svc.approvals.pending.return_value = [object()]
    res = c.handle(f"no, I said {MEANT}", source="typed")
    svc.approvals.answer.assert_not_called()
    assert res.corrected == MEANT
    assert svc.brain.chat.call_args.args == (MEANT,)


def test_the_not_x_comma_y_form(rich):
    c, svc = rich
    c.handle("open the terminal", source="typed")
    res = c.handle("not the terminal, what's the weather tomorrow", source="typed")
    assert res.corrected == "what's the weather tomorrow"
    svc.memory.log_correction.assert_called_once_with(
        "open the terminal", "what's the weather tomorrow")


@pytest.mark.parametrize("text,meant", [
    ("no, I said set a timer for ten minutes", "set a timer for ten minutes"),
    ("No. I said play some jazz.", "play some jazz"),
    ("I meant the calendar", "the calendar"),
    ("nope, I mean what's the weather", "what's the weather"),
    ("that's not what I said, I said call the dentist", "call the dentist"),
    ("not the terminal, the calendar", "the calendar"),
    ("I meant to ask about the weather", None),      # "meant to" is not a correction
    ("I mean it", None),
    ("no thanks", None),
    ("not now", None),                               # no comma: a plain sentence
    ("what's the weather", None),
])
def test_correction_kind(text, meant):
    assert correction_kind(text) == meant


# ------------------------------------------------------- vocab learning
def test_vocab_learning_is_off_by_default(rich):
    c, svc = rich
    c.handle("call pay rovi", source="typed")
    c.handle("no, I said call Peyrovi", source="typed")
    assert not PATHS.VOCAB_FILE.exists()


def test_vocab_learning_adds_the_new_name_when_enabled(rich):
    from jarvis.transcriber import DEFAULT_VOCAB, load_vocab
    c, svc = rich
    svc.assistant.data["corrections.learn_vocab"] = True
    c.handle("call pay rovi", source="typed")
    c.handle("no, I said call Peyrovi", source="typed")
    vocab = load_vocab()
    assert vocab.startswith(DEFAULT_VOCAB) and vocab.endswith(", Peyrovi")
    # a second correction with the same name does not duplicate it
    c.handle("no, I said call Peyrovi", source="typed")
    assert load_vocab().count("Peyrovi") == 1


@pytest.mark.parametrize("heard,meant,words", [
    ("call pay rovi", "call Peyrovi and Sarah", ["Peyrovi", "Sarah"]),
    ("what's on monday", "what's on Monday", []),          # a weekday, not a name
    ("", "Play some jazz", []),                            # a capitalised verb
    ("email bob", "email Bob", []),                        # the transcript had it
    ("", "the GB10 box", ["GB10"]),
])
def test_new_vocab_words(heard, meant, words):
    assert new_vocab_words(heard, meant) == words


# ------------------------------------------------------ brain generation
def test_a_cancelled_brain_job_stays_dead_after_the_next_one_starts(monkeypatch):
    """cancel() then chat(meant): the new acquire resets _cancelled, so the
    dead job's reply used to be remembered under the corrected text and
    spoken over the live answer. The job generation cannot be reset."""
    ctx = SimpleNamespace(exchanges=[])
    ctx.add_exchange = lambda u, j: ctx.exchanges.append((u, j))
    b = brain_mod.JarvisBrain(context=ctx, memory=None)
    gate = threading.Event()

    def slow(text, **kw):
        if text == "heard":
            gate.wait(5)
            return [("SPEAK", "stale reply")]
        return [("SPEAK", "fresh reply")]
    monkeypatch.setattr(b, "_chat_sync", slow)
    spoken = []

    t1 = b.chat("heard", callback=spoken.append)
    b.cancel()
    t2 = b.chat("meant", callback=spoken.append)
    assert t2 is not None, "the guard must be free after cancel()"
    t2.join(5)
    gate.set()
    t1.join(5)

    assert spoken == [[("SPEAK", "fresh reply")]]
    assert ctx.exchanges == [("meant", "fresh reply")]


def test_stale_is_generation_aware():
    b = brain_mod.JarvisBrain(None, None)
    assert b._acquire_busy()
    gen = b._job_gen
    assert b._stale(gen) is False
    b.cancel()
    assert b._stale(gen) is True            # cancelled
    assert b._acquire_busy()                # the next job resets _cancelled...
    assert b._stale(gen) is True            # ...but the old generation is stale
    assert b._stale(b._job_gen) is False
