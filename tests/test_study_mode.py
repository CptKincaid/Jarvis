"""'Teach me X': explain the material, then offer a quiz on the SAME
chunks (no second retrieval, so the questions are provably about what was
just explained).

Real modules throughout: a real chromadb DocsIndex over tests.test_docs's
FakeEmbed, a real Commander, a real FlashcardStore. No Ollama, no network.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
import jarvis.tools.docs as docs_mod
from jarvis.commander import (
    ASSISTANT_TIER1,
    REGISTRY,
    Commander,
    IntentClassifier,
    teach_kind,
)
from jarvis.config import CONFIG
from jarvis.tools.quiz import FlashcardStore, PREPARING_LINE
from tests.test_course_scope import BIOSENSORS_AUG_29, BIOSENSORS_AUG_30, SIGNALS_NOTE
from tests.test_docs import FakeEmbed


@pytest.fixture
def index(tmp_path):
    """The same two-course corpus test_course_scope indexes: two dated
    BIOSENSORS notes and one for SIGNALS AND SYSTEMS."""
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    (d / "biosensors-2026-08-29.md").write_text(BIOSENSORS_AUG_29)
    (d / "biosensors-2026-08-30.md").write_text(BIOSENSORS_AUG_30)
    (d / "signals-and-systems-2026-08-29.md").write_text(SIGNALS_NOTE)
    idx = docs_mod.DocsIndex([d], tmp_path / "index", embed=FakeEmbed())
    idx.reindex()
    return idx

# ------------------------------------------------------------- phrases
@pytest.mark.parametrize("text,topic", [
    ("teach me biosensors", "biosensors"),
    ("jarvis, teach me about chapter three", "chapter three"),
    ("teach me the nyquist criterion please", "nyquist criterion"),
    ("tutor me on signals and systems", "signals and systems"),
    ("walk me through biosensors like i'm new", "biosensors like i'm new"),
])
def test_teach_kind(text, topic):
    assert teach_kind(text) == topic


@pytest.mark.parametrize("text", [
    "teach me", "teach me a lesson", "teach", "teach me how to be patient",
])
def test_teach_kind_rejects(text):
    assert teach_kind(text) is None


def test_registry_and_tier1_carry_teach():
    names = [c.name for c in REGISTRY]
    # ahead of "workflow", the catch-all that matches everything
    assert names.index("teach me") < names.index("workflow")
    # after "quiz": "quiz me on X" must not be swallowed by the teach rung
    assert names.index("quiz") < names.index("teach me")
    assert "teach me" in {c.name for c in ASSISTANT_TIER1}


# ---------------------------------------------------------- commander
PAIRS = [{"question": "What does a biosensor couple?", "answer": "a recognition element"},
         {"question": "What does the Clark electrode measure?", "answer": "oxygen"}]


@pytest.fixture
def store(tmp_path):
    s = FlashcardStore(tmp_path / "flashcards.db")
    yield s
    s.close()


@pytest.fixture
def services(index, store):
    svc = SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
        context=MagicMock(), tts=MagicMock(), docs=index, flashcards=store,
        assistant=SimpleNamespace(
            get=lambda k, d=None: {"quiz.questions": 2, "canvas.token": ""}.get(k, d)),
    )
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock(), explain_calls=[], quiz_calls=[])
    svc.brain.explain_text = lambda text, name="": (
        svc.brain.explain_calls.append((text, name))
        or ("A biosensor couples a recognition element to a transducer, sir.",
            "A fuller paragraph about biosensors."))
    svc.brain.make_quiz = lambda text, n=5, topic="": (
        svc.brain.quiz_calls.append((text, n, topic)) or list(PAIRS[:n]))
    svc.replies = []
    svc.reply = lambda text, speak=True: svc.replies.append((text, speak))
    return svc


@pytest.fixture
def cmdr(services, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    return Commander(services)


def _teach(cmdr, services, text="teach me biosensors"):
    res = cmdr.handle(text, "typed")
    assert res.ack and res.done is False and res.speak
    return res


def test_teach_explains_from_the_scoped_chunks_and_offers_a_quiz(cmdr, services):
    _teach(cmdr, services)
    body, name = services.brain.explain_calls[0]
    # scoped: the other course's file is not in the study text
    assert "[biosensors-2026-08-29.md]" in body
    assert "signals-week-four.txt" not in body
    assert name == "biosensors"
    spoken, speak = services.replies[-1]
    assert speak and spoken.endswith(commander.TEACH_OFFER_LINE)
    assert cmdr._pending_teach is not None


def test_the_quiz_reuses_the_same_chunks_with_no_second_retrieval(cmdr, services,
                                                                  monkeypatch):
    _teach(cmdr, services)
    explained = services.brain.explain_calls[0][0]

    def boom(*a, **k):
        raise AssertionError("the teach offer re-retrieved the chunks")

    monkeypatch.setattr(commander, "course_chunks", boom)
    res = cmdr.handle("quiz me", "voice")
    assert res.handled and res.reply == PREPARING_LINE.format(topic="biosensors")
    body, n, topic = services.brain.quiz_calls[0]
    assert body == explained and topic == "biosensors" and n == 2
    assert cmdr._pending_quiz is not None
    assert cmdr._pending_teach is None                  # the offer is spent


def test_yes_takes_the_teach_offer_too(cmdr, services):
    _teach(cmdr, services)
    res = cmdr.handle("yes please", "voice")
    assert res.handled and services.brain.quiz_calls


def test_no_declines_and_clears_the_offer(cmdr, services):
    _teach(cmdr, services)
    res = cmdr.handle("no thank you", "voice")
    assert res.handled and not services.brain.quiz_calls
    assert cmdr._pending_teach is None


def test_a_new_subject_drops_the_offer_rather_than_quizzing(cmdr, services):
    _teach(cmdr, services)
    cmdr.handle("what's the time", "typed")
    assert cmdr._pending_teach is None and not services.brain.quiz_calls


def test_a_stale_offer_is_not_taken(cmdr, services, monkeypatch):
    _teach(cmdr, services)
    topic, chunks, _made = cmdr._pending_teach
    cmdr._pending_teach = (topic, chunks, 0.0)          # long ago
    monkeypatch.setattr(commander.time, "monotonic",
                        lambda: commander.OFFER_TTL_S + 10.0)
    cmdr.handle("yes", "voice")
    assert not services.brain.quiz_calls


def test_teach_on_an_unknown_topic_says_so(cmdr, services, monkeypatch):
    # Nothing in the folder is that close (the fake embedder's bag-of-words
    # vectors are generous; the real floor is MIN_TOPIC_SCORE).
    monkeypatch.setattr(docs_mod, "MIN_TOPIC_SCORE", 1.01)
    res = cmdr.handle("teach me quantum chromodynamics", "typed")
    assert res.handled
    assert "quantum chromodynamics" in services.replies[-1][0]
    assert not services.brain.explain_calls
    assert cmdr._pending_teach is None
