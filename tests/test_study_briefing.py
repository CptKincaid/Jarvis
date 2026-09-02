"""Exam-week study briefing: the deck line the morning briefing grows when
an exam is close, and the "shall we run ten now, sir?" offer the commander
answers.

Real modules: a real FlashcardStore over a tmp sqlite file, the real
build_briefing (with every network section switched off), a real Commander
for the offer rung. No Ollama, no network.
"""
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.tools.briefing import (OFFER_TTL_S, STUDY_NO_DECK_LINE, _exam_days,
                                   _study_section, build_briefing)
from jarvis.tools.quiz import NOTHING_DUE_LINE, FlashcardStore

# Off the REAL clock, not a frozen date. This file reads TWO clocks: the
# _study_section/build_briefing tests are handed NOW, but the commander
# tests below read back through cmdr.handle("yes") -> commander.py:5564
# store.due(limit=n, topic=topic), which takes no now= and falls through to
# time.time(). A frozen NOW writes cards at a fixed epoch the real clock
# eventually crosses: the box-3 deck was due 1788350699.0 = 2026-09-02
# 07:04:59, and test_a_deck_that_emptied_since_breakfast_says_so went red at
# breakfast that morning. Same rule as tests/test_notes_mail.py:872 -- the
# fixture must write with whichever clock the code under test reads.
NOW = datetime.now().astimezone()


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    yield


class Cfg:
    """assistant.json shape, every network section off so build_briefing
    exercises the coursework half only."""

    def __init__(self, **briefing):
        self.data = {"briefing": {
            "sections": {"weather": False, "calendar": False, "news": False,
                         "sports": False, "stocks": False, "canvas": True,
                         "study": True},
            "news_feeds": [], "sports_feeds": [], "stock_symbols": [], **briefing}}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def exam(days=2, course="BIOSENSORS", title="Midterm 1"):
    return {"course": course, "title": title, "when": NOW + timedelta(days=days),
            "kind": "exam", "all_day": False, "source": "canvas"}


@pytest.fixture
def store(tmp_path):
    s = FlashcardStore(tmp_path / "flashcards.db")
    yield s
    s.close()


def _deck(store, n, topic="BIOSENSORS", box=1, tag="a"):
    # tag keeps the questions distinct: add_cards dedupes on (question,
    # source), so two decks sharing both would be one deck.
    cards = store.add_cards([{"question": f"{tag}{i}?", "answer": f"x{i}"} for i in range(n)],
                            source=f"{topic.lower()}-notes.md", topic=topic,
                            now=NOW.timestamp() - 1)
    if box > 1:
        for card in cards:
            for _ in range(box - 1):
                store.record(card["id"], True, now=NOW.timestamp() - 1)
    return cards


# ------------------------------------------------------------ _exam_days
def test_exam_days_counts_whole_days_and_survives_junk():
    assert _exam_days(exam(days=0), NOW) == 0
    assert _exam_days(exam(days=6), NOW) == 6
    assert _exam_days({"when": "friday"}, NOW) is None
    assert _exam_days(None, NOW) is None


# --------------------------------------------------------- _study_section
def test_a_near_exam_grows_a_deck_line_and_an_offer(store):
    _deck(store, 4)                                  # box one: he keeps missing them
    _deck(store, 2, box=3, tag="b")                  # promoted: not due today
    line, offer = _study_section(Cfg(), exam(days=2), store, NOW)
    assert line.startswith("4 cards due on your BIOSENSORS deck, 4 of them in box one.")
    assert line.endswith("Shall we run four now, sir?")
    assert offer["course"] == "BIOSENSORS" and offer["n"] == 4 and offer["made_at"] > 0


def test_the_offer_is_capped_by_the_deck_and_by_the_setting(store):
    _deck(store, 30)
    _line, offer = _study_section(Cfg(), exam(days=1), store, NOW)
    assert offer["n"] == 10                          # study_offer_n default
    _line, offer = _study_section(Cfg(study_offer_n=3), exam(days=1), store, NOW)
    assert offer["n"] == 3


def test_no_offer_when_switched_off(store):
    _deck(store, 4)
    line, offer = _study_section(Cfg(study_offer=False), exam(days=1), store, NOW)
    assert line and "Shall we" not in line and offer == {}


def test_a_distant_exam_is_silent(store):
    _deck(store, 4)
    assert _study_section(Cfg(), exam(days=9), store, NOW) == ("", {})
    # study_days moves the horizon
    line, _ = _study_section(Cfg(study_days=10), exam(days=9), store, NOW)
    assert line


def test_an_exam_that_has_passed_is_silent(store):
    _deck(store, 4)
    assert _study_section(Cfg(), exam(days=-1), store, NOW) == ("", {})


def test_only_this_course_is_counted(store):
    _deck(store, 4, topic="BIOSENSORS")
    _deck(store, 7, topic="SIGNALS AND SYSTEMS")
    line, offer = _study_section(Cfg(), exam(days=1), store, NOW)
    assert line.startswith("4 cards due on your BIOSENSORS deck")
    assert offer["n"] == 4


def test_an_exam_with_no_course_cannot_be_scoped_so_says_nothing(store):
    _deck(store, 4)
    assert _study_section(Cfg(), exam(days=1, course=""), store, NOW) == ("", {})


def test_an_empty_deck_before_an_exam_says_so_and_offers_nothing(store):
    line, offer = _study_section(Cfg(), exam(days=1), store, NOW)
    assert line == STUDY_NO_DECK_LINE.format(course="BIOSENSORS") and offer == {}


def test_no_store_and_no_exam_are_both_silent(store):
    assert _study_section(Cfg(), exam(days=1), None, NOW) == ("", {})
    assert _study_section(Cfg(), None, store, NOW) == ("", {})


def test_a_store_that_raises_never_takes_the_briefing_down():
    class Broken:
        def due(self, **kw):
            raise RuntimeError("locked")

    assert _study_section(Cfg(), exam(days=1), Broken(), NOW) == ("", {})


def test_one_card_reads_as_one_card(store):
    _deck(store, 1)
    line, _ = _study_section(Cfg(), exam(days=1), store, NOW)
    assert line.startswith("One card due on your BIOSENSORS deck")


# --------------------------------------------------------- build_briefing
def test_build_briefing_carries_the_study_line_and_parks_the_offer(store, tmp_path):
    _deck(store, 4)
    parked = []
    sections, sheet = build_briefing(
        Cfg(), None, now=NOW, cache_path=tmp_path / "n.json",
        exam_lookup=lambda: exam(days=2), flashcards=store,
        park_offer=parked.append)
    assert sections["study"].startswith("4 cards due on your BIOSENSORS deck")
    assert f"Study: {sections['study']}" in sheet
    assert sections["exam"].startswith("Midterm 1 for BIOSENSORS")
    assert parked and parked[0]["course"] == "BIOSENSORS"


def test_the_study_section_can_be_switched_off_by_voice(store, tmp_path):
    _deck(store, 4)
    cfg = Cfg()
    cfg.data["briefing"]["sections"]["study"] = False
    parked = []
    sections, sheet = build_briefing(cfg, None, now=NOW, cache_path=tmp_path / "n.json",
                                     exam_lookup=lambda: exam(days=2), flashcards=store,
                                     park_offer=parked.append)
    assert sections["study"] == "" and "Study:" not in sheet and not parked
    assert sections["exam"]                          # the exam line is its own switch


def test_canvas_off_takes_the_study_section_with_it(store, tmp_path):
    _deck(store, 4)
    cfg = Cfg()
    cfg.data["briefing"]["sections"]["canvas"] = False
    sections, _sheet = build_briefing(cfg, None, now=NOW, cache_path=tmp_path / "n.json",
                                      exam_lookup=lambda: exam(days=2), flashcards=store)
    assert sections["study"] == "" and sections["exam"] == ""


def test_an_exam_lookup_that_raises_leaves_both_lines_empty(store, tmp_path):
    _deck(store, 4)

    def boom():
        raise RuntimeError("canvas down")

    sections, _sheet = build_briefing(Cfg(), None, now=NOW, cache_path=tmp_path / "n.json",
                                      exam_lookup=boom, flashcards=store)
    assert sections["study"] == "" and sections["exam"] == ""


def test_no_flashcard_store_is_not_an_error(tmp_path):
    sections, _sheet = build_briefing(Cfg(), None, now=NOW, cache_path=tmp_path / "n.json",
                                      exam_lookup=lambda: exam(days=2))
    assert sections["study"] == "" and sections["exam"]


# --------------------------------------------------------------- the rung
@pytest.fixture
def cmdr(store, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    svc = SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(), memory=MagicMock(),
                          context=MagicMock(), tts=MagicMock(), flashcards=store,
                          study_offer=None)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain = SimpleNamespace(think=MagicMock())
    return Commander(svc)


def _offer(cmdr, n=3, course="BIOSENSORS", made=None):
    import time as _t
    cmdr.services.study_offer = {"course": course, "n": n,
                                 "made_at": _t.time() if made is None else made}


def test_yes_deals_that_course_s_cards(cmdr, store):
    _deck(store, 5, topic="BIOSENSORS")
    _deck(store, 5, topic="SIGNALS AND SYSTEMS")
    _offer(cmdr, n=3)
    res = cmdr.handle("yes please", "voice")
    assert res.handled and cmdr._pending_quiz is not None
    assert cmdr._pending_quiz.topic == "BIOSENSORS"
    assert len(cmdr._pending_quiz.cards) == 3
    assert cmdr.services.study_offer is None


def test_no_declines_without_starting_anything(cmdr, store):
    _deck(store, 5)
    _offer(cmdr)
    res = cmdr.handle("no thanks", "voice")
    assert res.handled and cmdr._pending_quiz is None


def test_a_new_subject_drops_the_offer(cmdr, store):
    _deck(store, 5)
    _offer(cmdr)
    cmdr.handle("what's the time", "typed")
    assert cmdr.services.study_offer is None and cmdr._pending_quiz is None


def test_a_stale_offer_is_not_taken(cmdr, store):
    import time as _t
    _deck(store, 5)
    _offer(cmdr, made=_t.time() - OFFER_TTL_S - 10)
    cmdr.handle("yes", "voice")
    assert cmdr._pending_quiz is None


def test_a_deck_that_emptied_since_breakfast_says_so(cmdr, store):
    _deck(store, 2, topic="BIOSENSORS", box=3)       # promoted: not due today
    _offer(cmdr)
    res = cmdr.handle("yes", "voice")
    assert res.reply == NOTHING_DUE_LINE and cmdr._pending_quiz is None
