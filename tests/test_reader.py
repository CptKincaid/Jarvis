"""Tests for jarvis.reader (read aloud) and the commander's voice-I/O
Tier 1 commands: quiet, say again, pronounce, read aloud, continue."""
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
import jarvis.pronounce as pronounce
from jarvis.commander import (
    REGISTRY,
    Commander,
    IntentClassifier,
    continue_kind,
    quiet_kind,
    read_kind,
    repeat_kind,
)
from jarvis.config import CONFIG
from jarvis.pronounce import Pronunciations
from jarvis.reader import (
    CONTINUE_PROMPT,
    MAX_CHUNK_CHARS,
    ReadAloud,
    ReadResult,
    chunk_text,
    looks_like_text,
)


class FakeTTS:
    MAX_SPEAK_LENGTH = 500

    def __init__(self):
        self.spoken = []
        self.stopped = 0
        self.last_text = ""

    def speak(self, text):
        self.spoken.append(text)
        self.last_text = text

    def stop(self):
        self.stopped += 1

    def interrupt(self):
        self.stopped += 1
        return True

    def repeat_last(self):
        self.spoken.append(self.last_text)
        return bool(self.last_text)


def fake_run(stdout, returncode=0):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(stdout=stdout, returncode=returncode)
    run.calls = calls
    return run


# ---------------------------------------------------------- chunking
def test_chunk_text_respects_limit_and_keeps_order():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text)
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert len(chunks) > 1
    assert " ".join(chunks) == text


def test_chunk_text_splits_giant_sentence_at_spaces():
    text = "word " * 300
    chunks = chunk_text(text)
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)


def test_chunk_text_paragraph_breaks_and_empty():
    assert chunk_text("") == []
    assert chunk_text("First para.\n\nSecond para.") == ["First para. Second para."]


def test_looks_like_text():
    assert looks_like_text(b"hello\nworld\t!")
    assert not looks_like_text(b"\x00\x01\x02binary")


# ------------------------------------------------------------ reader
def test_read_text_in_parts_then_continue():
    tts = FakeTTS()
    r = ReadAloud(tts, max_part=60, max_chunk=30)
    text = ("Alpha sentence is here. Bravo sentence is here. "
            "Charlie sentence is here. Delta sentence is here.")
    res = r.read_text(text, label="the note")
    assert res.ok and res.remaining > 0
    assert tts.spoken[-1] == CONTINUE_PROMPT
    assert r.pending_chunks == res.remaining
    first_len = len(tts.spoken)
    res2 = r.continue_reading()
    assert res2.ok
    assert len(tts.spoken) > first_len
    # keep going until finished
    while r.pending_chunks:
        r.continue_reading()
    done = r.continue_reading()
    assert done.ok is False and "all of it" in done.message
    body = [s for s in tts.spoken if s != CONTINUE_PROMPT]
    assert " ".join(body) == text


def test_read_text_empty():
    r = ReadAloud(FakeTTS())
    res = r.read_text("   ")
    assert res.ok is False and "nothing to read" in res.message.lower()


def test_read_clipboard_uses_xclip():
    run = fake_run("Copied text here.")
    tts = FakeTTS()
    res = ReadAloud(tts, run=run).read_clipboard()
    assert res.ok and res.chunks == 1
    assert run.calls == [["xclip", "-selection", "clipboard", "-o"]]
    assert tts.spoken == ["Copied text here."]


def test_read_selection_empty_and_failed():
    res = ReadAloud(FakeTTS(), run=fake_run("")).read_selection()
    assert res.ok is False and "highlighted" in res.message
    res = ReadAloud(FakeTTS(), run=fake_run("x", returncode=1)).read_selection()
    assert res.ok is False


def test_read_file_text_binary_missing(tmp_path):
    tts = FakeTTS()
    r = ReadAloud(tts, search_dirs=[tmp_path])
    (tmp_path / "notes.md").write_text("# Title\n\nSome notes here.")
    res = r.read_file("notes.md")                     # relative → search dir
    assert res.ok and tts.spoken == ["# Title Some notes here."]
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02\x03" * 10)
    assert r.read_file(str(tmp_path / "blob.bin")).ok is False
    res = r.read_file("missing.txt")
    assert res.ok is False and "can't find" in res.message
    assert r.resolve_file("") is None


def test_stop_clears_pending_and_stops_tts():
    tts = FakeTTS()
    r = ReadAloud(tts, max_part=30, max_chunk=25)
    r.read_text("One sentence here. Two sentence here. Three sentence here.")
    assert r.pending_chunks > 0
    r.stop()
    assert r.pending_chunks == 0 and tts.stopped == 1


# ------------------------------------------------------- matchers
@pytest.mark.parametrize("text", [
    "jarvis, be quiet", "quiet", "shut up", "hush", "that's enough",
    "never mind", "stop talking", "Jarvis, stop reading.", "enough, jarvis",
    "stop", "Jarvis, stop.", "stop it",
])
def test_quiet_kind_matches(text):
    assert quiet_kind(text)


@pytest.mark.parametrize("text", [
    "stop recording", "stop listening", "quiet down the music",
    "be quiet about it", "what time is it", "stop the build",
])
def test_quiet_kind_rejects(text):
    assert not quiet_kind(text)


@pytest.mark.parametrize("text", [
    "say again", "jarvis, say that again", "repeat that", "come again?",
    "pardon?", "what was that", "once more", "repeat",
])
def test_repeat_kind_matches(text):
    assert repeat_kind(text)


def test_repeat_kind_rejects():
    assert not repeat_kind("repeat after me: hello")
    assert not repeat_kind("say hello")


@pytest.mark.parametrize("text,expected", [
    ("read the clipboard", ("clipboard", None)),
    ("jarvis, read my clipboard aloud", ("clipboard", None)),
    ("read what I copied", ("clipboard", None)),
    ("read clipboard", ("clipboard", None)),
    ("read this", ("selection", None)),
    ("read the selection to me", ("selection", None)),
    ("read the highlighted text", ("selection", None)),
    ("read file ~/notes.md", ("file", "~/notes.md")),
    ("Jarvis, read file /home/hunterp/Jarvis/README.md aloud",
     ("file", "/home/hunterp/Jarvis/README.md")),
    ("read aloud: the quick brown fox", ("text", "the quick brown fox")),
    ("read out loud the quick brown fox", ("text", "the quick brown fox")),
    ("read the quick brown fox aloud", ("text", "the quick brown fox")),
])
def test_read_kind(text, expected):
    assert read_kind(text) == expected


@pytest.mark.parametrize("text", [
    "read notes", "read my notes", "show notes", "what did i copy",
    "clipboard history", "read", "ready to go", "read me a story",
])
def test_read_kind_rejects_other_commands(text):
    assert read_kind(text) is None


def test_continue_kind():
    assert continue_kind("continue reading") and continue_kind("jarvis, go on")
    assert continue_kind("next part") and not continue_kind("continue the build")


def test_registry_order_for_voice_io():
    names = [c.name for c in REGISTRY]
    assert names.index("courtesy") < names.index("quiet") < names.index("workflow")
    assert names.index("read aloud") < names.index("clipboard")
    assert names.index("read aloud") < names.index("show notes")
    assert names.index("continue reading") < names.index("workflow")


# ------------------------------------------------------- commander
@pytest.fixture
def services():
    svc = types.SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=MagicMock(),
        memory=MagicMock(), context=MagicMock(), tts=FakeTTS(),
        reader=MagicMock(),
    )
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.reader.pending_chunks = 0
    return svc


@pytest.fixture
def cmdr(services, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG",
                        tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "auto_type", True)
    monkeypatch.setattr(CONFIG, "talkback", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    monkeypatch.setattr(pronounce, "_default",
                        Pronunciations(path=tmp_path / "pron.json"))
    return Commander(services)


def test_prefixed_quiet_interrupts_tts_and_reader(cmdr, services):
    res = cmdr.handle("jarvis, be quiet", "typed")
    assert res.handled and res.reply == "Very good, sir." and res.speak is False
    assert services.tts.stopped == 1
    services.reader.stop.assert_called_once()
    services.brain.think.assert_not_called()


def test_unprefixed_quiet_never_reaches_brain(cmdr, services):
    res = cmdr.handle("shut up", "typed")
    assert res.status == "Quiet"
    services.brain.think.assert_not_called()


def test_bare_stop_is_quiet_but_stop_recording_is_not(cmdr, services):
    res = cmdr.handle("stop", "typed")
    assert res.status == "Quiet" and services.tts.stopped == 1
    # Legacy voice phrase: stripped from the transcript, never a barge-in.
    res = cmdr.handle("stop recording", "typed")
    assert res.status == "No speech detected" and services.tts.stopped == 1


def test_repeat_replays_last_line(cmdr, services):
    services.tts.speak("Right here, sir.")
    res = cmdr.handle("jarvis, say again", "typed")
    assert res.reply == "Right here, sir." and res.speak is False
    assert services.tts.spoken == ["Right here, sir.", "Right here, sir."]
    assert res.status == "Repeating"


def test_repeat_with_talkback_off_only_shows(cmdr, services, monkeypatch):
    monkeypatch.setattr(CONFIG, "talkback", False)
    services.tts.speak("Yes, sir.")
    res = cmdr.handle("repeat that", "typed")
    assert res.reply == "Yes, sir." and services.tts.spoken == ["Yes, sir."]


def test_repeat_with_nothing_said(cmdr, services):
    res = cmdr.handle("say again", "typed")
    assert res.speak is True and "haven't said anything" in res.reply


def test_pronounce_command_persists_and_confirms(cmdr, services, tmp_path):
    res = cmdr.handle("jarvis, pronounce Peyrovi as pay-ROH-vee", "typed")
    assert res.speak is True and "Peyrovi" in res.reply
    assert pronounce.get().user_items() == {"Peyrovi": "pay-ROH-vee"}
    assert (tmp_path / "pron.json").exists()
    assert pronounce.apply("Mr Peyrovi") == "Mr pay-ROH-vee"
    res = cmdr.handle('pronounce "GB10" like "gee bee ten"', "typed")
    assert pronounce.apply("GB10") == "gee bee ten"


def test_read_clipboard_routes_to_reader_not_clipboard_snippet(cmdr, services):
    services.reader.read_clipboard.return_value = ReadResult(
        True, "Reading the clipboard: 2 chunk(s)", chunks=2)
    res = cmdr.handle("jarvis, read the clipboard", "typed")
    assert res.handled and res.status == "Reading the clipboard: 2 chunk(s)"
    assert res.reply is None                        # the reading is the reply
    services.reader.read_clipboard.assert_called_once()


def test_read_excuse_is_spoken(cmdr, services):
    services.reader.read_selection.return_value = ReadResult(
        False, "Nothing is highlighted, sir.")
    res = cmdr.handle("read this", "typed")        # unprefixed, jarvis mode
    assert res.speak is True and res.reply == "Nothing is highlighted, sir."
    services.brain.think.assert_not_called()


def test_read_file_and_inline_text(cmdr, services):
    services.reader.read_file.return_value = ReadResult(True, "Reading notes.md: 1 chunk(s)")
    cmdr.handle("jarvis read file ~/notes.md", "typed")
    services.reader.read_file.assert_called_once_with("~/notes.md")
    services.reader.read_text.return_value = ReadResult(True, "Reading that: 1 chunk(s)")
    cmdr.handle("jarvis, read aloud: hello there old friend", "typed")
    services.reader.read_text.assert_called_once_with("hello there old friend", label="that")


def test_read_notes_still_shows_notes(cmdr, services):
    services.memory.get_notes.return_value = []
    cmdr.handle("jarvis, read notes", "typed")
    services.memory.get_notes.assert_called_once()
    services.reader.read_text.assert_not_called()


def test_continue_reading_only_mid_reading(cmdr, services):
    services.reader.pending_chunks = 0
    cmdr.handle("continue", "typed")               # idle → brain, not reader
    services.reader.continue_reading.assert_not_called()
    services.brain.think.assert_called_once()
    services.reader.pending_chunks = 3
    services.reader.continue_reading.return_value = ReadResult(True, "Reading x: 2 chunk(s)")
    res = cmdr.handle("jarvis, continue reading", "typed")
    services.reader.continue_reading.assert_called_once()
    assert res.status == "Reading x: 2 chunk(s)"


def test_missing_reader_service_falls_through(services, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "i.json")
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    del services.reader
    c = Commander(services)
    c.handle("read the clipboard", "typed")
    services.brain.think.assert_called_once()       # graceful degradation
    assert commander.read_kind("read the clipboard") is not None
