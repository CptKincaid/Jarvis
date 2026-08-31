"""Sticky modes belong to the microphone (2026-08-30).

Dictation, lecture notes and an open quiz question used to swallow EVERY
utterance whatever its source: with notes open, `jarvis "what time is it"`
from a tmux shell was filed as a lecture line, a Discord message was graded
as a quiz answer, and in dictation mode a CLI turn was typed into whatever
window happened to be focused. The modes now act on source == "voice"
only; every other source routes normally.

Two escapes keep a terminal in control of a mode it cannot see: the
explicit end phrase is honoured from ANY source, and "note: ..." files a
deliberate lecture line. `app.open_modes_line()` (spoken by "status" /
"run diagnostics") names what is open.

Real commander, real notes store, real lecture file under tmp_path; the
routing assertions use the clock rung, which is answered locally with no
brain and no model.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.app as app_mod
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.lecture import END_LINE
from jarvis.tools.notes import NotesStore
from jarvis.tools.quiz import FlashcardStore, QuizSession


class Cfg:
    def __init__(self, folder):
        self.data = {"docs": {"paths": [str(folder)]},
                     "lecture": {"window_s": 20},
                     "canvas": {"base_url": "", "token": ""}}

    def get(self, dotted, default=None):
        obj = self.data
        for part in dotted.split("."):
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]
        return obj


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    return d


@pytest.fixture
def cmdr(folder, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    notes = NotesStore(tmp_path / "notes.db")
    svc = SimpleNamespace(assistant=Cfg(folder), notes=notes,
                          docs_index=SimpleNamespace(kick=lambda: True),
                          flashcards=FlashcardStore(tmp_path / "cards.db"),
                          desktop=SimpleNamespace(parse_action=lambda part: None),
                          tts=None)
    c = Commander(svc)
    yield c
    notes.close()


def _clock(res):
    """The clock rung answered it — i.e. the turn routed normally."""
    return res is not None and res.handled and res.status == "Clock"


def _note_file(folder):
    files = list((folder / "notes").glob("*.md"))
    return files[0].read_text() if files else ""


# ------------------------------------------------------------- lecture
@pytest.mark.parametrize("source", ["cli", "discord", "typed"])
def test_a_non_voice_turn_during_a_lecture_is_answered_not_filed(cmdr, folder,
                                                                 source):
    cmdr.handle("notes for biosensors", "typed")
    res = cmdr.handle("what time is it", source)
    assert _clock(res), res
    assert "what time is it" not in _note_file(folder)
    assert cmdr.lecture_course == "biosensors"      # still open for the mic


def test_a_spoken_line_is_still_filed(cmdr, folder):
    cmdr.handle("notes for biosensors", "typed")
    res = cmdr.handle("impedance is the ratio of voltage to current", "voice")
    assert res.status == "Noting: biosensors (1)"
    assert "impedance is the ratio" in _note_file(folder)


@pytest.mark.parametrize("source", ["cli", "discord"])
def test_the_end_phrase_closes_the_mode_from_any_source(cmdr, source):
    cmdr.handle("notes for biosensors", "typed")
    cmdr.handle("a spoken line", "voice")
    res = cmdr.handle("end notes", source)
    assert res.reply == END_LINE.format(n="one line", course="biosensors")
    assert cmdr.lecture_course is None


def test_a_note_prefix_files_a_deliberate_line_from_a_terminal(cmdr, folder):
    cmdr.handle("notes for biosensors", "typed")
    res = cmdr.handle("note: the demo is on friday", "cli")
    assert res.status == "Noting: biosensors (1)"
    assert "the demo is on friday" in _note_file(folder)
    # verbatim: the prefix is the escape hatch, so it never re-reads as the
    # end phrase or as a command
    res = cmdr.handle("note: end notes", "cli")
    assert res.status == "Noting: biosensors (2)" and cmdr.lecture_course


def test_a_ghost_lecture_flag_is_recovered_from_any_source(cmdr):
    """The flag outlived its file (a failed write): the recovery clears it
    and re-dispatches, and it must run for a box that only sees CLI turns."""
    cmdr.lecture_course = "ghost"
    cmdr._lecture = None
    res = cmdr.handle("what time is it", "cli")
    assert cmdr.lecture_course is None and _clock(res)


# ----------------------------------------------------------- dictation
def test_a_non_voice_turn_during_dictation_is_not_typed(cmdr, monkeypatch):
    typed = []
    monkeypatch.setattr(cmdr, "_type_raw", lambda t: typed.append(t))
    cmdr.handle("jarvis dictate")
    assert cmdr.dictation is True
    res = cmdr.handle("what time is it", "cli")
    assert typed == [] and _clock(res)
    cmdr.handle("hello world", "voice")
    assert typed == ["hello world "]


@pytest.mark.parametrize("source", ["cli", "discord"])
def test_end_dictation_works_from_any_source(cmdr, source):
    cmdr.handle("jarvis dictate")
    res = cmdr.handle("end dictation", source)
    assert res.reply == "Dictation mode: OFF" and cmdr.dictation is False


# ---------------------------------------------------------------- quiz
def _session(cmdr):
    cmdr._pending_quiz = QuizSession(
        [{"id": 1, "question": "What is impedance?", "answer": "V over I"},
         {"id": 2, "question": "Name a transducer", "answer": "thermistor"}],
        topic="biosensors")
    cmdr._pending_quiz.ask()
    return cmdr._pending_quiz


@pytest.mark.parametrize("source", ["cli", "discord", "typed"])
def test_a_non_voice_turn_is_not_graded_as_a_quiz_answer(cmdr, source):
    session = _session(cmdr)
    res = cmdr.handle("what time is it", source)
    assert _clock(res)
    assert cmdr._pending_quiz is session and session.asked == 0


def test_a_spoken_answer_is_still_graded(cmdr):
    session = _session(cmdr)
    res = cmdr.handle("v over i", "voice")
    assert session.asked == 1 and "Question 2" in res.reply


@pytest.mark.parametrize("source", ["cli", "discord"])
def test_the_quiz_stop_words_work_from_any_source(cmdr, source):
    _session(cmdr)
    res = cmdr.handle("stop the quiz", source)
    assert cmdr._pending_quiz is None and res.speak


# ------------------------------------------------------- status report
def _app(commander):
    a = object.__new__(app_mod.JarvisApp)
    a.commander = commander
    return a


def test_open_modes_line_names_what_is_open(cmdr, tmp_path):
    a = _app(cmdr)
    assert a.open_modes_line() == ""
    cmdr.handle("notes for biosensors", "typed")
    cmdr.handle("a line", "voice")
    assert a.open_modes_line() == "Lecture notes open for biosensors, 1 line."
    cmdr.dictation = True
    _session(cmdr)
    assert a.open_modes_line() == (
        "Lecture notes open for biosensors, 1 line, dictation mode on and "
        "a quiz open at question 1 of 2.")


def test_diagnostics_reports_the_open_modes(cmdr, monkeypatch):
    """"jarvis status" is the only way to notice a mode from a shell now
    that a CLI turn no longer lands in one."""
    a = _app(cmdr)
    a.tts = SimpleNamespace(engine="edge")
    a.recorder = SimpleNamespace(endpointer=None)
    a.speaker = None
    a._app_started = 0.0
    monkeypatch.setattr(app_mod, "_brain_model_name", lambda: "qwen", raising=False)
    cmdr.dictation = True
    assert "Dictation mode on." in a.diagnostics_text()


def test_open_modes_line_survives_a_commanderless_app():
    assert _app(None).open_modes_line() == ""
    assert _app(MagicMock(spec=[])).open_modes_line() == ""
