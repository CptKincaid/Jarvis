"""Nightly flashcards from lecture notes (jarvis/studycards.py).

The notes files are written exactly as jarvis/lecture.py writes them
(HEADER + LINE), so a change to that format breaks these tests rather than
silently producing cards about nothing.
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from jarvis import lecture as lecture_mod
from jarvis import studycards as sc
from jarvis.tools.quiz import FlashcardStore

NIGHT = datetime(2026, 9, 1, 3, 30)          # the small hours: the pass's window
EVENING = datetime(2026, 8, 31, 21, 0)


def write_notes(folder, course, day, lines):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{lecture_mod.slugify(course)}-{day}.md"
    text = lecture_mod.HEADER.format(course=course, date=day)
    for i, line in enumerate(lines):
        text += lecture_mod.LINE.format(time=f"09:{i:02d}", text=line)
    path.write_text(text)
    return path


class FakeQuiz:
    """The model seam: records what it was asked, answers with pairs."""

    def __init__(self, pairs=None, fail=False):
        self.pairs = pairs
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, text, n=5, topic=""):
        self.calls.append({"text": text, "n": n, "topic": topic})
        if self.fail:
            raise RuntimeError("ollama is down")
        if self.pairs is not None:
            return list(self.pairs)
        return [{"question": f"Q{i} about {topic}?", "answer": f"A{i}"}
                for i in range(n)]


@pytest.fixture
def notes(tmp_path):
    return tmp_path / "notes"


@pytest.fixture
def deck(tmp_path):
    store = FlashcardStore(tmp_path / "cards.db")
    yield store
    store.close()


def cards(tmp_path, notes, deck, quiz=None, now=NIGHT, **cfg):
    settings = {"study_cards.enabled": True, "study_cards.per_course": 3,
                "study_cards.max_courses": 3, "study_cards.run_before_hour": 5,
                "study_cards.max_age_days": 7, "study_cards.min_lines": 3}
    settings.update(cfg)
    conf = SimpleNamespace(get=lambda k, d=None: settings.get(k, d))
    return sc.NightlyCards(cfg=conf, store=deck, make_quiz=quiz or FakeQuiz(),
                           state_path=tmp_path / "studycards.json",
                           now=lambda: now, notes_dir=lambda: notes)


# ----------------------------------------------------------------- reading
def test_read_notes_takes_the_course_from_the_header(notes):
    """The deck is filtered by the course as Canvas titles it; the file
    name only carries the slug."""
    path = write_notes(notes, "BIOSENSORS", "2026-08-31",
                       ["the transducer converts the biological signal",
                        "selectivity comes from the recognition layer",
                        "drift is the enemy"])
    note = sc.read_notes(path)
    assert note["course"] == "BIOSENSORS"
    assert note["date"] == "2026-08-31"
    assert len(note["lines"]) == 3
    assert note["lines"][0].startswith("the transducer")


def test_a_header_only_file_is_not_notes(notes):
    notes.mkdir(parents=True)
    path = notes / "biosensors-2026-08-31.md"
    path.write_text(lecture_mod.HEADER.format(course="BIOSENSORS",
                                              date="2026-08-31"))
    assert sc.read_notes(path) == {}


def test_newest_notes_keeps_one_file_per_course(notes):
    write_notes(notes, "BIOSENSORS", "2026-08-29", ["a", "b", "c"])
    newest = write_notes(notes, "BIOSENSORS", "2026-08-31", ["d", "e", "f"])
    write_notes(notes, "SIGNALS AND SYSTEMS", "2026-08-30", ["g", "h", "i"])
    found = sc.newest_notes(notes, NIGHT, 7)
    assert [e["path"] for e in found if e["slug"] == "biosensors"] == [newest]
    assert len(found) == 2


def test_notes_older_than_the_window_are_left_alone(notes):
    write_notes(notes, "BIOSENSORS", "2026-07-01", ["a", "b", "c"])
    assert sc.newest_notes(notes, NIGHT, 7) == []


def test_a_missing_notes_folder_is_not_an_error(tmp_path):
    assert sc.newest_notes(tmp_path / "nope", NIGHT, 7) == []


# -------------------------------------------------------------- the pass
def test_the_pass_turns_each_courses_notes_into_cards(tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a a a", "b b b", "c c c"])
    write_notes(notes, "SYSTEMS PHYSIOLOGY", "2026-08-31", ["d", "e", "f"])
    quiz = FakeQuiz()
    n = cards(tmp_path, notes, deck, quiz)
    record = n.run_pass()
    assert {m["course"] for m in record["made"]} == {"BIOSENSORS",
                                                    "SYSTEMS PHYSIOLOGY"}
    assert deck.count() == 6                     # per_course=3, two courses
    assert {c["topic"] for c in quiz.calls} == {"BIOSENSORS", "SYSTEMS PHYSIOLOGY"}


def test_the_cards_carry_the_course_the_briefing_filters_by(tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    cards(tmp_path, notes, deck).run_pass()
    due = deck.due(limit=50, now=NIGHT.timestamp(), topic="BIOSENSORS")
    assert len(due) == 3
    assert due[0]["source"].startswith(sc.SOURCE_PREFIX)


def test_a_file_is_carded_once(tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    quiz = FakeQuiz()
    n = cards(tmp_path, notes, deck, quiz)
    n.run_pass()
    n.run_pass()
    assert len(quiz.calls) == 1
    assert deck.count() == 3


def test_a_lecture_continued_after_the_pass_is_looked_at_again(tmp_path, notes,
                                                               deck):
    """The fingerprint is the line count, so notes that grew are new work
    -- and the store's own de-duplication stops the deck doubling."""
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    quiz = FakeQuiz()
    n = cards(tmp_path, notes, deck, quiz)
    n.run_pass()
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c", "d", "e"])
    n.run_pass()
    assert len(quiz.calls) == 2
    assert deck.count() == 3                     # same questions, same source


def test_a_thin_file_is_never_carded(tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["only one line"])
    quiz = FakeQuiz()
    assert cards(tmp_path, notes, deck, quiz).run_pass()["made"] == []
    assert quiz.calls == []


def test_only_max_courses_are_carded_in_a_night(tmp_path, notes, deck):
    for name in ("ONE", "TWO", "THREE", "FOUR"):
        write_notes(notes, name, "2026-08-31", ["a", "b", "c"])
    n = cards(tmp_path, notes, deck, **{"study_cards.max_courses": 2})
    assert len(n.run_pass()["made"]) == 2


def test_a_lent_gpu_defers_the_whole_pass(tmp_path, notes, deck):
    """Skipped, never queued: the notes are still there tomorrow night, and
    a 26B question-writing run must never sit in front of a trainer."""
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    quiz = FakeQuiz()
    n = cards(tmp_path, notes, deck, quiz)
    n._lent = lambda: True
    record = n.run_pass()
    assert record["skipped"] == "lent to a trainer"
    assert quiz.calls == [] and deck.count() == 0
    assert not (tmp_path / "studycards.json").exists()   # not recorded: retried


def test_a_busy_model_defers_the_pass(tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    n = cards(tmp_path, notes, deck)
    n._busy = lambda: True
    assert n.run_pass()["skipped"] == "busy"


def test_a_model_that_raises_costs_one_course_not_the_night(tmp_path, notes,
                                                            deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    n = cards(tmp_path, notes, deck, FakeQuiz(fail=True))
    record = n.run_pass()
    assert record["made"] == [] and deck.count() == 0
    assert record["day"] == "2026-09-01"          # the pass itself completed


def test_a_course_the_model_had_nothing_for_is_not_retried_all_night(
        tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    quiz = FakeQuiz(pairs=[])
    n = cards(tmp_path, notes, deck, quiz)
    n.run_pass()
    n.run_pass()
    assert len(quiz.calls) == 1


# ------------------------------------------------------------ scheduling
def test_due_only_in_the_small_hours(tmp_path, notes, deck):
    assert cards(tmp_path, notes, deck, now=NIGHT).due() is True
    assert cards(tmp_path, notes, deck, now=EVENING).due() is False


def test_one_pass_a_night(tmp_path, notes, deck):
    n = cards(tmp_path, notes, deck)
    assert n.due() is True
    n.run_pass()
    assert n.due() is False


def test_the_switch_stops_it_dead(tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    n = cards(tmp_path, notes, deck, **{"study_cards.enabled": False})
    assert n.due() is False
    assert n.tick() is None
    n.start()
    assert n.running is False


def test_state_survives_a_restart(tmp_path, notes, deck):
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    quiz = FakeQuiz()
    cards(tmp_path, notes, deck, quiz).run_pass()
    cards(tmp_path, notes, deck, quiz).run_pass()
    assert len(quiz.calls) == 1
    state = json.loads((tmp_path / "studycards.json").read_text())
    assert state["done"]["biosensors-2026-08-31.md"] == "3"


def test_a_corrupt_state_file_is_a_fresh_start(tmp_path, notes, deck):
    (tmp_path / "studycards.json").write_text("{not json")
    write_notes(notes, "BIOSENSORS", "2026-08-31", ["a", "b", "c"])
    assert cards(tmp_path, notes, deck).run_pass()["made"]


def test_start_and_stop_are_joinable(tmp_path, notes, deck):
    n = cards(tmp_path, notes, deck)
    n.start()
    assert n.running is True
    n.stop()
    assert n.running is False


def test_without_a_deck_nothing_happens(tmp_path, notes):
    n = sc.NightlyCards(cfg=None, store=None, make_quiz=FakeQuiz(),
                        state_path=tmp_path / "s.json",
                        now=lambda: NIGHT, notes_dir=lambda: notes)
    assert n.run_pass()["skipped"] == "no deck"
