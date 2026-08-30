"""Spelled names in both directions: "my advisor's name is spelled
P-E-Y-R-O-V-I, say it pay-ROH-vee" and "add X to your vocabulary".

Real Commander and Router, mocked services (the test_corrections
pattern); the pronunciation table and every vocabulary file live under
tmp_path. No model, no audio, no network.
"""
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.pronounce as pronounce
from jarvis import vocab
from jarvis.commander import Commander, IntentClassifier, spelled_word
from jarvis.config import CONFIG, PATHS
from jarvis.pronounce import Pronunciations
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
def cmdr(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(Commander, "FEEDBACK_LOG",
                        tmp_path / "feedback.jsonl", raising=False)
    monkeypatch.setattr(PATHS, "VOCAB_FILE", tmp_path / "voice_vocab.txt")
    monkeypatch.setattr(PATHS, "NAMES_FILE", tmp_path / "voice_names.txt")
    monkeypatch.setattr(PATHS, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pronounce, "_default",
                        Pronunciations(path=tmp_path / "pron.json"))
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", True), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    vocab.clear_cache()
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=MagicMock(),
        assistant=Cfg(), timekeeper=MagicMock(), notes=MagicMock(),
        approvals=MagicMock(), claude=MagicMock(),
        conversation=SimpleNamespace(forget_exchange=MagicMock(
            return_value=True)))
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.approvals.pending.return_value = []
    svc.claude.active_project = "jarvis"
    svc.brain.local_line.return_value = "Right away, sir."
    svc.router = Router(svc.assistant, classify=lambda t: ("local", 0.2))
    yield Commander(svc)
    vocab.clear_cache()


def names_file():
    try:
        return PATHS.NAMES_FILE.read_text()
    except FileNotFoundError:
        return ""


# --------------------------------------------------------- normalisation
@pytest.mark.parametrize("run,word", [
    ("P-E-Y-R-O-V-I", "Peyrovi"),
    ("p e y r o v i", "Peyrovi"),
    ("P. E. Y. R. O. V. I", "Peyrovi"),
    ("P, E, Y, R, O, V, I", "Peyrovi"),
    ("PEYROVI", "Peyrovi"),          # Whisper glued the letters
    ("pey rovi", ""),                # mixed tokens: never guess a boundary
    ("p", ""),                       # one letter is not a name
    ("", ""),
])
def test_spelled_word_normalisation(run, word):
    assert spelled_word(run) == word


# ------------------------------------------------------- the full command
def test_full_sentence_teaches_both_directions(cmdr):
    res = cmdr.handle("jarvis, my advisor's name is spelled P-E-Y-R-O-V-I, "
                      "say it pay-ROH-vee", "typed")
    assert res.handled and res.speak
    assert "P-E-Y-R-O-V-I" in res.reply and "Peyrovi" in res.reply
    assert names_file() == "Peyrovi\n"
    assert pronounce.get().user_items() == {"Peyrovi": "pay-ROH-vee"}
    assert pronounce.apply("Mr Peyrovi") == "Mr pay-ROH-vee"


def test_dotted_letters_normalise(cmdr):
    res = cmdr.handle("jarvis, my name is spelled a. m. b. r. o. s. e.",
                      "typed")
    assert res.status == "Spelled Ambrose"
    assert names_file() == "Ambrose\n"


def test_no_say_clause_touches_only_the_names_file(cmdr):
    res = cmdr.handle("jarvis, the name is spelled P E Y R O V I", "typed")
    assert res.status == "Spelled Peyrovi"
    assert names_file() == "Peyrovi\n"
    assert pronounce.get().user_items() == {}


def test_glued_uppercase_word(cmdr):
    res = cmdr.handle("jarvis, the name is spelled PEYROVI, "
                      "pronounce it pay-ROH-vee", "typed")
    assert res.status == "Spelled Peyrovi"
    assert pronounce.get().user_items() == {"Peyrovi": "pay-ROH-vee"}


def test_unprefixed_in_jarvis_mode(cmdr):
    res = cmdr.handle("my advisor's name is spelled P-E-Y-R-O-V-I", "typed")
    assert res.status == "Spelled Peyrovi"
    assert names_file() == "Peyrovi\n"


def test_mixed_tokens_do_not_store_a_guess(cmdr):
    cmdr.handle("jarvis, my advisor's name is spelled pay rovi", "typed")
    assert names_file() == ""


# ------------------------------------------------------ add to vocabulary
def test_add_to_vocabulary_preserves_casing(cmdr):
    res = cmdr.handle("jarvis, add Librespot to your vocabulary", "typed")
    assert res.handled and "Librespot" in res.reply
    assert names_file() == "Librespot\n"


def test_add_duplicate_is_acknowledged_not_duplicated(cmdr):
    cmdr.handle("jarvis, add Librespot to your vocabulary", "typed")
    res = cmdr.handle("jarvis, add librespot to my vocab", "typed")
    assert "already" in res.reply
    assert names_file() == "Librespot\n"


# --------------------------------------------------- reaching the prompt
def test_prompt_picks_the_name_up_immediately(cmdr):
    assert "Peyrovi" not in vocab.build_prompt()   # warm the 60 s cache
    cmdr.handle("jarvis, the name is spelled P-E-Y-R-O-V-I", "typed")
    assert "Peyrovi" in vocab.build_prompt()       # add_name busted it
