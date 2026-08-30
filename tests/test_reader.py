"""Tests for jarvis.reader (read aloud) and the commander's voice-I/O
Tier 1 commands: quiet, say again, pronounce, read aloud, continue."""
import threading as _threading
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


# ------------------------------------------------ steering: skip/back/pause
# 2026-08-30. The only control over a reading used to be "quiet", which
# dropped the whole queue. The reader now owns a cursor: TTS.speak() hands
# back each chunk's done-event, so it knows which chunk is in the air, and
# it keeps its own history for "back". No TTS ids, no per-item bookkeeping
# in the TTS beyond skip_current().


class SteerTTS(FakeTTS):
    """FakeTTS whose speak() returns a done-event the test controls (as the
    real TTS does) and which counts skip_current() calls."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.skips = 0

    def speak(self, text):
        super().speak(text)
        ev = _threading.Event()
        self.events.append((text, ev))
        return ev

    def skip_current(self):
        self.skips += 1
        return True

    def finish(self, n):
        """Pretend the first ``n`` queued utterances have played."""
        for _, ev in self.events[:n]:
            ev.set()


THREE = "Alpha part is here. Bravo part is here. Charlie part is here."


def steer_reader(max_part=1000):
    tts = SteerTTS()
    r = ReadAloud(tts, max_part=max_part, max_chunk=22)
    res = r.read_text(THREE, label="the note")
    assert res.ok and res.chunks == 3 and tts.spoken == [
        "Alpha part is here.", "Bravo part is here.", "Charlie part is here."]
    return r, tts


def test_active_only_while_a_chunk_is_in_the_air():
    r, tts = steer_reader()
    assert r.active and not r.paused
    tts.finish(2)
    assert r.active                                   # Charlie still queued
    tts.finish(3)
    assert not r.active                               # the part has played out
    assert r.pending_chunks == 0


def test_active_covers_the_continue_prompt_and_pending_parts_do_not_count():
    tts = SteerTTS()
    r = ReadAloud(tts, max_part=45, max_chunk=22)
    r.read_text(THREE)
    assert tts.spoken[-1] == CONTINUE_PROMPT and r.pending_chunks == 1
    tts.finish(2)                                     # two chunks done
    assert r.active                                   # the prompt is speaking
    tts.finish(3)
    assert not r.active, "waiting for 'continue' is idle: Spotify gets its words back"


def test_skip_cuts_only_the_current_chunk_and_keeps_the_queue():
    r, tts = steer_reader()
    res = r.skip()
    assert res.ok and tts.skips == 1
    assert tts.stopped == 0                           # never the whole queue
    assert len(tts.spoken) == 3                       # nothing re-queued
    tts.finish(3)
    assert r.skip().ok is False                       # nothing in the air


def test_back_re_reads_the_previous_chunk_then_carries_on():
    r, tts = steer_reader()
    tts.finish(1)                                     # Alpha done, Bravo playing
    res = r.back()
    assert res.ok and tts.stopped == 1
    assert tts.spoken[3:] == ["Alpha part is here.", "Bravo part is here.",
                              "Charlie part is here."]
    assert r.active


def test_back_at_the_start_restarts_the_current_chunk():
    r, tts = steer_reader()
    res = r.back()                                    # nothing spoken yet
    assert res.ok
    assert tts.spoken[3:] == ["Alpha part is here.", "Bravo part is here.",
                              "Charlie part is here."]


def test_back_after_skip_returns_to_the_skipped_chunk():
    r, tts = steer_reader()
    r.skip()
    tts.finish(1)                                     # the skipped one is 'done'
    r.back()
    assert tts.spoken[3] == "Alpha part is here."


def test_pause_holds_the_rest_and_go_on_resumes_from_the_cut_chunk():
    r, tts = steer_reader()
    tts.finish(1)                                     # Alpha done, Bravo playing
    res = r.pause()
    assert res.ok and res.message == "Paused, sir." and tts.stopped == 1
    assert r.paused and r.active                      # still owns the words
    assert r.pending_chunks == 2                      # Bravo (restart) + Charlie
    assert r.pause().ok is False                      # "Already paused, sir."
    assert r.resume().ok
    assert not r.paused
    assert tts.spoken[3:] == ["Bravo part is here.", "Charlie part is here."]


def test_continue_reading_also_lifts_a_pause():
    r, tts = steer_reader()
    r.pause()
    assert r.continue_reading().ok and not r.paused


def test_resume_when_not_paused_is_none_so_go_on_keeps_its_meaning():
    r, tts = steer_reader()
    assert r.resume() is None


def test_paused_skip_and_back_move_the_cursor_without_speaking():
    r, tts = steer_reader()
    tts.finish(1)
    r.pause()                                         # pending: Bravo, Charlie
    spoken_before = len(tts.spoken)
    assert r.skip().ok and r.pending_chunks == 1      # Bravo dropped
    assert r.back().ok and r.pending_chunks == 2      # ... and back again
    assert r.back().ok and r.pending_chunks == 3      # Alpha too
    assert r.back().ok is False                       # "We're at the start, sir."
    assert len(tts.spoken) == spoken_before and r.paused
    r.resume()
    assert tts.spoken[spoken_before:] == [
        "Alpha part is here.", "Bravo part is here.", "Charlie part is here."]


def test_stop_and_a_new_reading_clear_the_pause_and_the_history():
    r, tts = steer_reader()
    tts.finish(1)
    r.pause()
    r.stop()
    assert not r.paused and not r.active and r.pending_chunks == 0
    r.read_text("Fresh start here.")
    assert r.back().ok                                # restarts, no stale history
    assert tts.spoken[-1] == "Fresh start here."


def test_a_tts_without_done_events_still_reads_but_cannot_be_steered():
    tts = FakeTTS()                                   # speak() returns None
    r = ReadAloud(tts, max_part=1000, max_chunk=22)
    assert r.read_text(THREE).ok
    assert not r.active                               # no handle, no claim on the words


# ----------------------------------------------- commander: context gating
@pytest.mark.parametrize("text,kind", [
    ("skip", "skip"), ("jarvis, skip that", "skip"), ("skip ahead", "skip"),
    ("back", "back"), ("go back", "back"), ("jarvis go back a bit", "back"),
    ("say that again", "back"), ("previous", "back"),
    ("pause", "pause"), ("hold on", "pause"), ("hang on", "pause"),
    ("wait a second", "pause"), ("pause reading", "pause"),
    ("go on", "resume"), ("carry on", "resume"), ("resume", "resume"),
    ("continue reading", "resume"), ("keep going", "resume"),
])
def test_read_control_kind(text, kind):
    assert commander.read_control_kind(text) == kind


@pytest.mark.parametrize("text", [
    "skip to the next track", "pause the music", "go back to the terminal",
    "play some jazz", "continue the deployment", "back up the database",
])
def test_read_control_kind_rejects_longer_commands(text):
    assert commander.read_control_kind(text) is None


def test_read_control_is_tier_one_after_continue_reading():
    names = [c.name for c in REGISTRY]
    assert names.index("continue reading") < names.index("read control") < names.index("workflow")
    assert "read control" in [c.name for c in commander.ASSISTANT_TIER1]


def test_transport_words_keep_their_meaning_when_nothing_is_read(cmdr, services):
    """Idle reader: "pause"/"skip" go on to the router (Spotify's transport
    tool lives behind the brain) and "jarvis, go back" is the window switch."""
    services.reader.active = False
    services.context.get_last_window.return_value = "Terminal"
    cmdr.handle("pause", "typed")
    services.reader.pause.assert_not_called()
    services.brain.think.assert_called_with("pause")                   # on to the router
    cmdr.handle("skip", "typed")
    services.reader.skip.assert_not_called()
    services.brain.think.assert_called_with("skip")
    res = cmdr.handle("jarvis, go back", "typed")
    services.reader.back.assert_not_called()
    assert res.status == "Back to Terminal"                            # the window "go back"


def test_transport_words_steer_an_active_reading(cmdr, services):
    services.reader.active = True
    services.reader.skip.return_value = ReadResult(True, "Skipped")
    services.reader.back.return_value = ReadResult(True, "Reading the note: 2 chunk(s)")
    services.reader.pause.return_value = ReadResult(True, "Paused, sir.")
    services.reader.resume.return_value = ReadResult(True, "Reading the note: 1 chunk(s)")

    res = cmdr.handle("skip", "voice")                # one word, unprefixed, by voice
    services.reader.skip.assert_called_once()
    assert res.handled and res.speak is False         # the next chunk is the confirmation
    services.desktop.handle_action.assert_not_called()
    services.brain.think.assert_not_called()

    res = cmdr.handle("jarvis, go back", "voice")
    services.reader.back.assert_called_once()
    services.context.get_last_window.assert_not_called()   # not the window switch

    res = cmdr.handle("pause", "typed")
    services.reader.pause.assert_called_once()
    assert res.reply == "Paused, sir." and res.speak is True

    res = cmdr.handle("go on", "typed")
    services.reader.resume.assert_called_once()
    assert res.status == "Reading the note: 1 chunk(s)"


def test_go_on_while_reading_unpaused_still_means_the_next_part(cmdr, services):
    services.reader.active = True
    services.reader.paused = False
    services.reader.resume.return_value = None        # not paused
    services.reader.pending_chunks = 2
    services.reader.continue_reading.return_value = ReadResult(True, "Reading x: 2 chunk(s)")
    res = cmdr.handle("go on", "typed")
    services.reader.continue_reading.assert_called_once()
    assert res.status == "Reading x: 2 chunk(s)"


def test_steering_excuses_are_spoken(cmdr, services):
    services.reader.active = True
    services.reader.back.return_value = ReadResult(False, "We're at the start, sir.")
    res = cmdr.handle("back", "typed")
    assert res.reply == "We're at the start, sir." and res.speak is True


def test_quiet_still_ends_the_reading_and_frees_the_words(cmdr, services):
    services.reader.active = True
    cmdr.handle("jarvis, quiet", "typed")
    services.reader.stop.assert_called_once()
