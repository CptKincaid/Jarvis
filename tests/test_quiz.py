"""Quiz mode over the documents index (spec 16): the Leitner flashcard
store, the string grader, the session, topic retrieval from a real
chromadb store through the fake embedder, the commander's quiz rung
(ask -> answer -> grade -> next -> tally) and the two brain calls.

No Ollama: the embedder is tests.test_docs.FakeEmbed, the generation and
grading calls are fakes on services.brain, brain._http is a recorder.
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.brain as brain
import jarvis.commander as commander
import jarvis.tools.docs as docs_mod
import jarvis.tools.quiz as quiz
from jarvis.commander import (
    ASSISTANT_TIER1,
    REGISTRY,
    Commander,
    IntentClassifier,
    quiz_kind,
    quiz_stop_kind,
    review_kind,
)
from jarvis.config import CONFIG, PATHS
from jarvis.tools.docs import INDEXING_LINE, EmbedError, topic_chunks
from jarvis.tools.quiz import (
    CORRECT_LINES,
    NO_CARDS_LINE,
    NO_DOCS_LINE,
    NO_TOPIC_LINE,
    NOTHING_DUE_LINE,
    PREPARING_LINE,
    QUIZ_DONE_LINE,
    SKIP_LINE,
    UNSURE_LINE,
    WRONG_LINE,
    FlashcardStore,
    QuizSession,
    grade_by_string,
    normalise,
    study_text,
)
from tests.test_docs import RECIPE, SYLLABUS, FakeEmbed

CHAPTERS = """Physics Notes

Chapter 1: Motion
Velocity is the rate of change of position. Acceleration is the rate of
change of velocity.

Chapter 2: Forces
Newton's second law: force equals mass times acceleration.

Chapter 3: Energy
Kinetic energy is one half m v squared. Potential energy is m g h. Energy
is conserved in a closed system.
"""


# --------------------------------------------------------------- store
@pytest.fixture
def store(tmp_path):
    s = FlashcardStore(tmp_path / "flashcards.db")
    yield s
    s.close()


def test_default_db_lives_under_memory_dir():
    assert quiz.default_db_path().parent == PATHS.MEMORY_DIR
    assert "aiws_trainer/jarvis_memory" not in str(quiz.default_db_path())


def test_add_cards_dedupes_and_starts_in_box_one(store):
    now = 1_000_000.0
    cards = store.add_cards([{"question": "What is V?", "answer": "IR"},
                             {"question": "what is v?", "answer": "I R"},
                             {"question": "", "answer": "x"}], source="notes.md", now=now)
    assert [c["question"] for c in cards] == ["What is V?", "What is V?"]
    assert store.count() == 1 and cards[0]["box"] == 1 and cards[0]["due"] == now


def test_leitner_promote_and_demote(store):
    now = 1_000_000.0
    card = store.add_cards([{"question": "q", "answer": "a"}], now=now)[0]
    up = store.record(card["id"], True, now=now)
    assert up["box"] == 2 and up["due"] == now + quiz.DAY_S
    up = store.record(card["id"], True, now=now)
    assert up["box"] == 3 and up["due"] == now + 3 * quiz.DAY_S
    for _ in range(5):
        up = store.record(card["id"], True, now=now)
    assert up["box"] == quiz.BOXES and up["due"] == now + 14 * quiz.DAY_S
    down = store.record(card["id"], False, now=now + 5)
    assert down["box"] == 1 and down["due"] == now + 5
    assert down["seen"] == 8 and down["correct"] == 7


def test_due_orders_lowest_box_first_and_filters_by_topic(store):
    now = 1_000_000.0
    a, b, c = store.add_cards([{"question": "a?", "answer": "1"},
                               {"question": "b?", "answer": "2"},
                               {"question": "c?", "answer": "3"}],
                              source="physics.pdf", topic="chapter three", now=now)
    store.record(b["id"], True, now=now)           # box 2, due tomorrow
    store.record(c["id"], False, now=now)          # box 1, due now
    store.record(a["id"], True, now=now)
    store.record(a["id"], True, now=now)           # box 3, due in 3 days
    assert [x["question"] for x in store.due(now=now)] == ["c?"]
    assert [x["question"] for x in store.due(now=now + 2 * quiz.DAY_S)] == ["c?", "b?"]
    assert store.due(now=now + 30 * quiz.DAY_S, topic="chapter") and \
        store.due(now=now + 30 * quiz.DAY_S, topic="chemistry") == []
    st = store.stats(now=now)
    assert st == {"total": 3, "due": 1, "boxes": {1: 1, 2: 1, 3: 1}}


# ------------------------------------------------------------- grading
@pytest.mark.parametrize("expected,given,verdict", [
    ("forty percent", "forty percent", True),
    ("40%", "I think it's forty percent", True),
    ("forty percent", "twenty percent", False),
    ("October 14", "the fourteenth of october", None),      # a model question
    ("October 14", "October fourteen", None),
    ("mass times acceleration", "force is mass times acceleration", True),
    ("mass times acceleration", "the acceleration", None),
    ("Newton", "Einstein", False),
    ("the mitochondria", "mitochondrion", None),
    ("", "anything", None),
    ("7", "seven", True),
    ("12 volts", "12", True),
])
def test_grade_by_string(expected, given, verdict):
    assert grade_by_string(expected, given) is verdict


def test_normalise_drops_filler_and_plurals():
    assert normalise("Um, I think it's the two proteins") == ["2", "protein"]


# ------------------------------------------------------------- session
def test_session_asks_settles_and_scores():
    s = QuizSession([{"id": 1, "question": "q1", "answer": "a"},
                     {"id": 2, "question": "q2", "answer": "b"}], topic="t")
    assert s.ask(now=10.0) == "Question 1: q1" and s.asked_at == 10.0
    assert not s.stale(now=10.0 + quiz.ANSWER_WINDOW_S - 1)
    assert s.stale(now=10.0 + quiz.ANSWER_WINDOW_S + 1)
    assert s.score_line() == quiz.QUIZ_STOPPED_EARLY_LINE
    s.settle(True)
    assert s.score_line() == quiz.QUIZ_STOPPED_LINE.format(right=1, asked=1)
    assert s.ask() == "Question 2: q2"
    s.settle(None)                                   # skipped: no tally change
    assert s.finished and s.ask() == "" and s.current is None
    assert s.score_line() == QUIZ_DONE_LINE.format(right=1, total=2)
    assert s.results == [(1, True), (2, None)]


def test_study_text_names_the_file_and_caps():
    chunks = [{"name": "a.md", "text": "x " * 100}, {"name": "b.md", "text": "y " * 100}]
    out = study_text(chunks, cap=150)
    assert out.startswith("[a.md] x x") and "[b.md]" not in out


# ------------------------------------------------------- topic_chunks
@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    (d / "syllabus.md").write_text(SYLLABUS)
    (d / "recipe.txt").write_text(RECIPE)
    (d / "physics_notes.txt").write_text(CHAPTERS)
    return d


@pytest.fixture
def index(tmp_path, folder):
    embed = FakeEmbed()
    idx = docs_mod.DocsIndex([folder], tmp_path / "index", embed=embed)
    idx.reindex()
    idx.fake_embed = embed
    return idx


def test_names_and_chunks_of(index):
    assert index.names() == ["physics_notes.txt", "recipe.txt", "syllabus.md"]
    chunks = index.chunks_of("syllabus.md")
    assert chunks and [c["chunk"] for c in chunks] == list(range(len(chunks)))
    assert all(c["name"] == "syllabus.md" for c in chunks)
    assert index.chunks_of("nope.pdf") == []


def test_topic_by_file_name_needs_no_embedding(index):
    n = len(index.fake_embed.calls)
    hits = topic_chunks(index, "the syllabus", k=3)
    assert hits and all(h["name"] == "syllabus.md" for h in hits)
    assert len(index.fake_embed.calls) == n


def test_topic_chapter_heading_starts_at_the_chapter(index, monkeypatch):
    # One chunk per paragraph so each chapter is its own chunk (chunk_text
    # is looked up on the module at call time; its size default is not).
    monkeypatch.setattr(docs_mod, "chunk_text", lambda text, *a, **k: [
        p.strip() for p in text.split("\n\n") if p.strip()])
    path = next(p for p in index.scan() if p.name == "physics_notes.txt")
    index._index_file(index.collection(), path, (1.0, 1))     # replaces its chunks
    hits = topic_chunks(index, "chapter three", k=2)
    assert hits and "Kinetic energy" in hits[0]["text"]
    assert all(h["name"] == "physics_notes.txt" for h in hits)
    hits = topic_chunks(index, "the physics notes chapter 2", k=1)
    assert hits and "Newton" in hits[0]["text"]


def test_topic_falls_back_to_the_embedding_query_in_reading_order(index):
    hits = topic_chunks(index, "how many eggs go in the pancakes", k=3)
    assert hits and "recipe.txt" in {h["name"] for h in hits}
    assert hits == sorted(hits, key=lambda h: (h["name"], h["chunk"]))
    assert all(h["score"] >= docs_mod.MIN_TOPIC_SCORE for h in hits)
    assert topic_chunks(index, "", k=3) == []


def test_topic_hits_under_the_score_floor_are_dropped(index, monkeypatch):
    monkeypatch.setattr(docs_mod, "MIN_TOPIC_SCORE", 1.01)     # nothing is that close
    assert topic_chunks(index, "how many eggs go in the pancakes", k=3) == []


def test_topic_raises_only_when_the_embedder_is_down(index):
    index.fake_embed.down = True
    assert topic_chunks(index, "syllabus", k=2)      # name match: no embed call
    with pytest.raises(EmbedError):
        topic_chunks(index, "something about nothing", k=2)


def test_make_tools_parks_the_index_on_services(folder, tmp_path):
    cfg = {"docs": {"paths": [str(folder)], "index_dir": str(tmp_path / "i")}}
    services = SimpleNamespace(docs=None)
    docs_mod.make_tools(cfg, services, embed=FakeEmbed())
    assert isinstance(services.docs, docs_mod.DocsIndex)
    assert not (tmp_path / "i").exists()             # still lazy


# ------------------------------------------------------------ phrases
@pytest.mark.parametrize("text,topic", [
    ("quiz me on chapter three", "chapter three"),
    ("jarvis, quiz me about the syllabus please", "syllabus"),
    ("test me on my physics notes", "physics notes"),
    ("drill me over the biosensors handout.", "biosensors handout"),
])
def test_quiz_kind(text, topic):
    assert quiz_kind(text) == topic


def test_quiz_kind_rejects():
    assert quiz_kind("quiz night is on thursday") is None
    assert quiz_kind("test the build") is None


@pytest.mark.parametrize("text", [
    "review my flashcards", "let's review the flash cards", "flashcards",
    "start a flashcard review", "go through my cards", "practice my deck",
])
def test_review_kind(text):
    assert review_kind(text)


@pytest.mark.parametrize("text", [
    "stop the quiz", "end the quiz", "stop quizzing me", "that's enough questions",
    "jarvis, enough of the flashcards", "no more questions",
])
def test_quiz_stop_kind(text):
    assert quiz_stop_kind(text)


def test_registry_and_tier1_carry_the_quiz_commands():
    names = [c.name for c in REGISTRY]
    assert names.index("quiz") < names.index("workflow")
    assert names.index("stop quiz") < names.index("workflow")
    tier1 = [c.name for c in ASSISTANT_TIER1]
    assert {"quiz", "review flashcards", "stop quiz"} <= set(tier1)


# ---------------------------------------------------------- commander
PAIRS = [{"question": "What share of the grade is homework?", "answer": "forty percent"},
         {"question": "When is the midterm exam?", "answer": "October 14"},
         {"question": "How much does late homework lose per day?", "answer": "ten percent"}]


@pytest.fixture
def services(index, store):
    svc = SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
        context=MagicMock(), tts=MagicMock(), docs=index, flashcards=store,
        assistant=SimpleNamespace(get=lambda k, d=None: {"quiz.questions": 3}.get(k, d)),
    )
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock(), quiz_calls=[], grade_calls=[])
    svc.brain.make_quiz = lambda text, n=5, topic="": (
        svc.brain.quiz_calls.append((text, n, topic)) or list(PAIRS[:n]))
    svc.brain.grade_answer = lambda q, a, g: (
        svc.brain.grade_calls.append((q, a, g)) or (True, "The fourteenth, quite right."))
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


def _start(cmdr, services, text="quiz me on the syllabus"):
    # "typed": an unprefixed voice command meets the intent gate first, as
    # every Tier 1 command does (the hotword consumes the wake word). The
    # ANSWERS below are "voice" -- the quiz rung must sit ahead of that gate,
    # which calls "forty percent" background chat.
    res = cmdr.handle(text, "typed")
    assert res.ack and res.done is False and res.speak
    assert res.reply == PREPARING_LINE.format(topic="syllabus")
    assert services.replies[-1] == ("Question 1: What share of the grade is homework?", True)
    return res


def test_quiz_round_trip_grades_and_files_cards(cmdr, services, store):
    _start(cmdr, services)
    text, n, topic = services.brain.quiz_calls[0]
    assert n == 3 and topic == "syllabus" and "[syllabus.md]" in text
    assert store.count() == 3 and cmdr._pending_quiz is not None

    # 1. right by the string match: no model call, box two, next question
    res = cmdr.handle("forty percent", "voice")
    assert res.speak and res.done and not res.ack
    assert any(res.reply.startswith(line) for line in CORRECT_LINES)
    assert res.reply.endswith("Question 2: When is the midterm exam?")
    assert services.brain.grade_calls == []
    # 2. unclear -> the model grades it
    res = cmdr.handle("the fourteenth of october", "voice")
    assert services.brain.grade_calls == [("When is the midterm exam?", "October 14",
                                           "the fourteenth of october")]
    assert res.reply.endswith("Question 3: How much does late homework lose per day?")
    # 3. wrong: the answer is named, the tally closes the quiz
    res = cmdr.handle("five percent", "voice")
    assert res.reply.startswith(WRONG_LINE.format(answer="ten percent"))
    assert res.reply.endswith(QUIZ_DONE_LINE.format(right=2, total=3))
    assert cmdr._pending_quiz is None and res.status == "Quiz finished"
    boxes = sorted((c["question"][:4], c["box"]) for c in
                   [store.get(i) for i in (1, 2, 3)])
    assert boxes == [("How ", 1), ("What", 2), ("When", 2)]
    services.brain.think.assert_not_called()


def test_skip_reveals_the_answer_and_moves_on(cmdr, services, store):
    _start(cmdr, services)
    res = cmdr.handle("I don't know", "voice")
    assert res.reply.startswith(SKIP_LINE.format(answer="forty percent"))
    assert "Question 2" in res.reply
    assert store.get(1)["seen"] == 0                 # ungraded: no Leitner move


def test_stop_ends_with_the_tally_and_quiet_ends_silently(cmdr, services):
    _start(cmdr, services)
    cmdr.handle("forty percent", "voice")
    res = cmdr.handle("stop the quiz", "voice")
    assert res.reply == quiz.QUIZ_STOPPED_LINE.format(right=1, asked=1) and res.speak
    assert cmdr._pending_quiz is None
    _start(cmdr, services)
    res = cmdr.handle("quiet", "voice")
    assert res.speak is False and cmdr._pending_quiz is None
    # no quiz open: "stop the quiz" is nobody's command here -> the model
    res = cmdr.handle("stop the quiz", "typed")
    services.brain.think.assert_called_once()


def test_grader_failure_names_the_answer_without_a_leitner_move(cmdr, services, store):
    services.brain.grade_answer = lambda q, a, g: None
    _start(cmdr, services)
    res = cmdr.handle("in the autumn sometime", "voice")   # overlaps nothing -> wrong
    assert res.reply.startswith(WRONG_LINE.format(answer="forty percent"))
    res = cmdr.handle("October fourteen", "voice")         # unclear, model silent
    assert res.reply.startswith(UNSURE_LINE.format(answer="October 14"))
    assert store.get(2)["seen"] == 0


def test_a_stale_question_is_dropped_and_the_text_routes(cmdr, services):
    _start(cmdr, services)
    cmdr._pending_quiz.asked_at = time.time() - quiz.ANSWER_WINDOW_S - 1
    cmdr.handle("what time is it", "typed")
    assert cmdr._pending_quiz is None


def test_a_new_quiz_replaces_the_old(cmdr, services):
    _start(cmdr, services)
    res = cmdr.handle("quiz me on the recipe", "typed")
    assert res.reply == PREPARING_LINE.format(topic="recipe")
    assert cmdr._pending_quiz is not None and cmdr._pending_quiz.topic == "recipe"


def test_review_asks_the_due_cards(cmdr, services, store):
    res = cmdr.handle("review my flashcards", "typed")
    assert res.reply == NO_CARDS_LINE and res.speak
    _start(cmdr, services)
    cmdr.handle("forty percent", "voice")        # box 2, due tomorrow
    cmdr.handle("nonsense", "voice")             # wrong: due now
    cmdr.handle("ten percent", "voice")          # box 2, due tomorrow (quiz over)
    res = cmdr.handle("review my flashcards", "typed")
    assert res.reply.startswith("One card due, sir. Question 1: When is the midterm")
    assert cmdr._pending_quiz is not None and cmdr._pending_quiz.total == 1
    res = cmdr.handle("october 14", "voice")
    assert res.reply.endswith(QUIZ_DONE_LINE.format(right=1, total=1))
    res = cmdr.handle("review my flashcards", "typed")
    assert res.reply == NOTHING_DUE_LINE


def test_quiz_excuses(cmdr, services, index, monkeypatch):
    # off-topic: the fallback query's hits all sit under the score floor
    monkeypatch.setattr(docs_mod, "MIN_TOPIC_SCORE", 1.01)
    res = cmdr.handle("quiz me on quantum chromodynamics", "typed")
    assert res.ack and services.replies[-1] == (
        NO_TOPIC_LINE.format(topic="quantum chromodynamics"), True)
    monkeypatch.setattr(docs_mod, "MIN_TOPIC_SCORE", 0.3)
    services.brain.make_quiz = lambda text, n=5, topic="": []
    cmdr.handle("quiz me on the syllabus", "typed")
    assert services.replies[-1][0] == quiz.NO_QUESTIONS_LINE.format(topic="syllabus")
    index.fake_embed.down = True
    cmdr.handle("quiz me on something unindexed", "typed")
    assert services.replies[-1][0] == quiz.INDEX_DOWN_LINE
    del services.docs
    res = cmdr.handle("quiz me on the syllabus", "typed")
    assert res.reply == NO_DOCS_LINE and res.speak


def test_quiz_on_an_empty_store_kicks_the_index(cmdr, services, tmp_path, folder):
    fresh = docs_mod.DocsIndex([folder], tmp_path / "fresh", embed=FakeEmbed())
    services.docs = fresh
    cmdr.handle("quiz me on the syllabus", "typed")
    assert services.replies[-1][0] == INDEXING_LINE
    fresh.wait(5)
    assert fresh.document_count() == 3


def test_flashcard_store_is_built_lazily_under_memory_dir(cmdr, services):
    del services.flashcards
    assert cmdr._flashcards is None
    cmdr.handle("review my flashcards", "typed")
    assert cmdr._flashcards is not None
    assert cmdr._flashcards.db_path == PATHS.MEMORY_DIR / "flashcards.db"
    cmdr._flashcards.close()


# ---------------------------------------------------------------- brain
class FakeHttp:
    def __init__(self, content):
        self.calls = []
        self.content = content

    def __call__(self, path, payload=None, timeout=None):
        self.calls.append((path, payload, timeout))
        if isinstance(self.content, Exception):
            raise self.content
        return {"message": {"role": "assistant", "content": self.content}}


def test_make_quiz_payload_and_parsing(monkeypatch):
    brain.reset_static_prompt()
    fake = FakeHttp(json.dumps({"questions": [
        {"question": " What is V? ", "answer": "I  R"}, {"question": "", "answer": "x"},
        "junk", {"question": "q2", "answer": "a2"}, {"question": "q3", "answer": "a3"}]}))
    monkeypatch.setattr(brain, "_http", fake)
    out = brain.make_quiz("study text " * 50, n=2, topic="ohm's law")
    assert out == [{"question": "What is V?", "answer": "I R"}, {"question": "q2", "answer": "a2"}]
    path, payload, timeout = fake.calls[0]
    assert "tools" not in payload and payload["format"] == brain.QUIZ_FORMAT
    assert payload["options"]["num_predict"] == 220 and timeout == brain.QUIZ_TIMEOUT_S
    assert "ohm's law" in payload["messages"][1]["content"]
    assert payload["messages"][0]["content"] == brain.static_system()
    monkeypatch.setattr(brain, "_http", FakeHttp(brain.OllamaDown("x")))
    assert brain.make_quiz("text") == []
    assert brain.make_quiz("") == []


def test_grade_answer_payload_and_fallbacks(monkeypatch):
    fake = FakeHttp('{"correct": true, "note": "Quite right, sir. Extra sentence."}')
    monkeypatch.setattr(brain, "_http", fake)
    assert brain.grade_answer("q", "October 14", "the fourteenth") == (True, "Quite right, sir.")
    path, payload, timeout = fake.calls[0]
    assert "tools" not in payload and payload["format"] == brain.GRADE_FORMAT
    assert payload["options"]["temperature"] == 0.0 and timeout == brain.GRADE_TIMEOUT_S
    assert "Expected answer: October 14" in payload["messages"][1]["content"]
    monkeypatch.setattr(brain, "_http", FakeHttp('{"note": "no verdict"}'))
    assert brain.grade_answer("q", "a", "g") is None
    monkeypatch.setattr(brain, "_http", FakeHttp("garbage"))
    assert brain.grade_answer("q", "a", "g") is None


def test_the_quiz_config_defaults_exist():
    from jarvis.assistant_config import DEFAULTS
    # window_s: how long the mic stays open for the ANSWER (app._capture_window)
    assert DEFAULTS["quiz"] == {"questions": 5, "chunks": 6, "window_s": 15}
    assert commander._int_setting(SimpleNamespace(_svc=lambda n: None), "quiz.questions", 5) == 5
