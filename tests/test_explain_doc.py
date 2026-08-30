"""Explain this PDF (spec 11): "explain the biosensors lab handout" gives a
two-sentence spoken lead and a fuller card, "read it to me" then reads the
same document through the continue-reading flow; read-aloud accepts .pdf
and .docx by way of tools.docs.extract_text and resolves spoken names
against the docs folders.

Real modules throughout (ReadAloud with a fake TTS, the real Commander,
brain.explain_text against a fake _http); pdftotext is not run -- the PDF
seam (_pdf_text) is replaced so the tests hold with poppler absent.
"""
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.app as app_mod
import jarvis.brain as brain
import jarvis.commander as commander
import jarvis.tools.docs as docs_mod
from jarvis.commander import (
    ASSISTANT_TIER1,
    EXPLAIN_ACK_LINE,
    EXPLAIN_FAIL_LINE,
    NO_DOCUMENT_LINE,
    NO_SUCH_DOCUMENT_LINE,
    READ_OFFER_LINE,
    REGISTRY,
    Commander,
    IntentClassifier,
    explain_kind,
    read_kind,
)
from jarvis.config import CONFIG
from jarvis.events import JarvisReply, bus
from jarvis.reader import (
    CONTINUE_PROMPT,
    NO_TEXT_LINE,
    ReadAloud,
    unwrap_text,
)

PDF_TEXT = ("Biosensors Lab Handout\n\nA biosensor couples a bio-\nlogical element "
            "to a transducer.\nThe lab report is due on October 14 and is worth\n"
            "twenty percent of the grade.\n\fPage 2\n\nBring gloves.\n")


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


def _docx(path: Path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w='
           '"http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", xml)


@pytest.fixture
def docs_dir(tmp_path, monkeypatch):
    d = tmp_path / "Jarvis Docs"
    d.mkdir()
    (d / "Biosensors_Lab_Handout.pdf").write_bytes(b"%PDF-1.4 fake")
    (d / "empty.pdf").write_bytes(b"%PDF-1.4 fake")
    _docx(d / "CS101_syllabus.docx", ["CS 101 Syllabus", "Homework is forty percent."])
    (d / "notes").mkdir()
    (d / "notes" / "week3_notes.md").write_text("# Week 3\nOhm's law: V = IR.\n")
    (d / "recipe.txt").write_text("Whisk two cups of flour.")

    def fake_pdf(path):
        return PDF_TEXT if path.name.startswith("Biosensors") else ""
    monkeypatch.setattr(docs_mod, "_pdf_text", fake_pdf)
    return d


@pytest.fixture
def reader(docs_dir, tmp_path):
    tts = FakeTTS()
    r = ReadAloud(tts, search_dirs=[tmp_path / "nowhere", docs_dir])
    r.tts = tts
    return r


# ------------------------------------------------------------- reader
def test_unwrap_joins_hard_wraps_and_keeps_paragraphs():
    out = unwrap_text(PDF_TEXT)
    assert "biological element to a transducer. The lab report is due on October 14" in out
    assert "\n\n" in out and "\f" not in out
    assert out.count("\n\n") >= 2                    # the page break became a paragraph


def test_read_file_accepts_pdf_and_docx(reader):
    res = reader.read_file("Biosensors_Lab_Handout.pdf")
    assert res.ok and res.chunks >= 1
    spoken = " ".join(reader.tts.spoken)
    assert "biological element to a transducer" in spoken
    assert "October 14" in spoken
    reader.tts.spoken.clear()
    res = reader.read_file("CS101_syllabus.docx")
    assert res.ok and "Homework is forty percent." in " ".join(reader.tts.spoken)


def test_read_file_with_no_extractable_text_is_an_excuse(reader):
    res = reader.read_file("empty.pdf")
    assert not res.ok and res.message == NO_TEXT_LINE.format(name="empty.pdf")
    assert reader.tts.spoken == []


@pytest.mark.parametrize("spoken,expected", [
    ("the biosensors lab handout", "Biosensors_Lab_Handout.pdf"),
    ("biosensors", "Biosensors_Lab_Handout.pdf"),
    ("cs101 syllabus", "CS101_syllabus.docx"),
    ("the syllabus", "CS101_syllabus.docx"),
    ("week three notes", "week3_notes.md"),          # one folder down, number word
    ("recipe.txt", "recipe.txt"),                     # the exact name still wins
])
def test_resolve_document_matches_spoken_names(reader, spoken, expected):
    path = reader.resolve_document(spoken)
    assert path is not None and path.name == expected


def test_resolve_document_refuses_a_different_document(reader):
    assert reader.resolve_document("the thermodynamics textbook") is None
    assert reader.resolve_document("") is None


def test_long_document_reads_in_parts(reader, docs_dir, monkeypatch):
    monkeypatch.setattr(docs_mod, "_pdf_text",
                        lambda p: " ".join(f"Sentence number {i} is here." for i in range(300)))
    res = reader.read_file("Biosensors_Lab_Handout.pdf")
    assert res.ok and res.remaining > 0
    assert reader.tts.spoken[-1] == CONTINUE_PROMPT
    reader.tts.spoken.clear()
    assert reader.continue_reading().ok and reader.tts.spoken


# ---------------------------------------------------------- explain_kind
@pytest.mark.parametrize("text,expected", [
    ("explain the biosensors lab handout", ("strong", "biosensors lab handout")),
    ("jarvis, summarize the cs101 syllabus for me", ("strong", "cs101 syllabus")),
    ("summarise my lecture notes", ("strong", "lecture notes")),
    ("walk me through the biosensors pdf", ("strong", "biosensors pdf")),
    ("give me a summary of report.pdf", ("strong", "report.pdf")),
    ("explain the lab report", ("weak", "lab report")),
    ("summarize that paper", ("weak", "paper")),
    ("explain chapter 7 bankruptcy", ("weak", "chapter 7 bankruptcy")),   # falls through
])
def test_explain_kind(text, expected):
    assert explain_kind(text) == expected


@pytest.mark.parametrize("text", [
    "summarize my inbox", "explain the theory of relativity", "explain",
    "summarize what mark said", "tell me about the syllabus",
])
def test_explain_kind_leaves_the_rest_to_the_model(text):
    assert explain_kind(text) is None


def test_read_kind_document_forms():
    assert read_kind("read the document") == ("document", None)
    assert read_kind("read that handout to me") == ("document", None)
    assert read_kind("jarvis, read the whole document") == ("document", None)
    assert read_kind("read it to me") == ("selection", None)   # diverted only when fresh
    assert read_kind("read the clipboard") == ("clipboard", None)


def test_registry_places_explain_and_quiz_before_the_catch_alls():
    names = [c.name for c in REGISTRY]
    assert names.index("continue reading") < names.index("explain document") \
        < names.index("workflow")
    assert names.index("explain document") < names.index("show notes")
    assert names.index("explain document") < names.index("clipboard")
    tier1 = [c.name for c in ASSISTANT_TIER1]
    assert "explain document" in tier1 and "quiz" in tier1


# ------------------------------------------------------------ commander
@pytest.fixture
def services(reader):
    svc = SimpleNamespace(
        desktop=MagicMock(), workflows=MagicMock(), brain=SimpleNamespace(),
        memory=MagicMock(), context=MagicMock(), tts=reader.tts, reader=reader,
    )
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.context.get_last_window.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.brain.think = MagicMock()
    svc.brain.explain_text = lambda text, name="": (
        f"It's the {name}, sir. The report is due October 14.",
        "The handout explains biosensors. The lab report is due on October 14 and "
        "is worth twenty percent. Bring gloves.")
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


@pytest.fixture
def cards():
    seen = []
    bus.subscribe(JarvisReply, seen.append)
    yield seen
    bus.unsubscribe(JarvisReply, seen.append)


def test_explain_acks_then_delivers_lead_and_card(cmdr, services, cards):
    res = cmdr.handle("jarvis, explain the biosensors lab handout", "typed")
    assert res.handled and res.ack and res.done is False and res.speak
    assert res.reply == EXPLAIN_ACK_LINE.format(name="Biosensors Lab Handout")
    # the worker ran inline: the lead (plus the offer) went through services.reply
    assert len(services.replies) == 1
    text, speak = services.replies[0]
    assert speak and text.startswith("It's the Biosensors Lab Handout, sir.")
    assert text.endswith(READ_OFFER_LINE)
    bus.drain()
    card = [e for e in cards if not e.speak]
    assert card and "Biosensors_Lab_Handout.pdf" in card[0].text \
        and "Bring gloves" in card[0].text
    assert services.reader.tts.spoken == []          # nothing read aloud yet
    services.brain.think.assert_not_called()


def test_read_it_to_me_after_explain_reads_the_document(cmdr, services):
    cmdr.handle("explain the biosensors handout", "typed")        # unprefixed, jarvis mode
    res = cmdr.handle("read it to me", "typed")
    assert res.handled and res.status.startswith("Reading Biosensors_Lab_Handout.pdf")
    assert "biological element" in " ".join(services.reader.tts.spoken)
    services.brain.think.assert_not_called()


def test_read_the_document_with_nothing_explained(cmdr, services, monkeypatch):
    res = cmdr.handle("jarvis, read the document", "typed")
    assert res.reply == NO_DOCUMENT_LINE and res.speak
    # "read it" with no fresh document is still the selection
    monkeypatch.setattr(commander, "_xclip", lambda *a, **k: "", raising=False)
    services.reader._run = lambda *a, **k: SimpleNamespace(stdout="", returncode=1)
    res = cmdr.handle("read it", "typed")
    assert res.reply == "Nothing is highlighted, sir."


def test_a_stale_document_is_not_read_it(cmdr, services):
    cmdr.handle("explain the biosensors handout", "typed")
    path, when = cmdr._last_document
    cmdr._last_document = (path, when - commander.LAST_DOCUMENT_S - 1)
    services.reader._run = lambda *a, **k: SimpleNamespace(stdout="", returncode=1)
    res = cmdr.handle("read it to me", "typed")
    assert res.reply == "Nothing is highlighted, sir."


def test_unknown_strong_name_is_an_excuse_weak_falls_through(cmdr, services):
    res = cmdr.handle("explain the thermodynamics handout", "typed")
    assert res.reply == NO_SUCH_DOCUMENT_LINE.format(name="thermodynamics handout")
    assert res.speak and services.replies == []
    res = cmdr.handle("explain the lab report", "typed")     # weak word, no such file
    services.brain.think.assert_called_once()                # the model's question


def test_explain_failure_is_spoken_not_silent(cmdr, services):
    services.brain.explain_text = lambda text, name="": ("", "")
    cmdr.handle("explain the biosensors handout", "typed")
    assert services.replies == [(EXPLAIN_FAIL_LINE, True)]
    # and the document is still on hand for "read it to me"
    assert cmdr.handle("read it to me", "typed").status.startswith("Reading")


def test_explain_without_a_model_offers_to_read(cmdr, services):
    del services.brain.explain_text
    res = cmdr.handle("explain the biosensors handout", "typed")
    assert res.speak and "read it to me" in res.reply
    assert cmdr.handle("read it to me", "typed").status.startswith("Reading")


def test_explain_empty_document(cmdr, services):
    res = cmdr.handle("explain empty.pdf", "typed")
    assert res.reply == NO_TEXT_LINE.format(name="empty.pdf")


def test_explain_ack_is_not_graded_as_a_reading(cmdr, services):
    """The ack carries ack=True so the app treats it as a filler, not the
    answer (the turn stays open for the worker's reply)."""
    res = cmdr.handle("summarize the cs101 syllabus", "typed")
    assert res.ack and res.done is False
    assert services.replies and "cs101 syllabus".lower() in services.replies[0][0].lower()


# ----------------------------------------------------------------- brain
class FakeHttp:
    def __init__(self, content):
        self.calls = []
        self.content = content

    def __call__(self, path, payload=None, timeout=None):
        self.calls.append((path, payload, timeout))
        if isinstance(self.content, Exception):
            raise self.content
        return {"message": {"role": "assistant", "content": self.content}}


def test_explain_text_is_one_tool_free_json_call(monkeypatch):
    brain.reset_static_prompt()
    fake = FakeHttp('{"lead": "It is the **lab handout**, sir. The report is due October 14. '
                    'And a third sentence.", "summary": "Five sentences of detail here."}')
    monkeypatch.setattr(brain, "_http", fake)
    lead, summary = brain.explain_text("x" * 20_000, name="lab handout")
    path, payload, timeout = fake.calls[0]
    assert path == "/api/chat" and timeout == brain.EXPLAIN_TIMEOUT_S
    assert "tools" not in payload and payload["format"] == brain.EXPLAIN_FORMAT
    assert payload["options"]["num_ctx"] == brain.NUM_CTX
    assert payload["messages"][0]["content"] == brain.static_system()
    user = payload["messages"][1]["content"]
    assert len(user) < brain.EXPLAIN_MAX_CHARS + 1200 and "it goes on" in user
    assert lead == "It is the lab handout, sir. The report is due October 14."
    assert summary == "Five sentences of detail here."


def test_explain_text_survives_a_dead_model(monkeypatch):
    monkeypatch.setattr(brain, "_http", FakeHttp(brain.OllamaDown("refused")))
    assert brain.explain_text("some text") == ("", "")
    monkeypatch.setattr(brain, "_http", FakeHttp("not json at all"))
    assert brain.explain_text("some text") == ("", "")
    assert brain.explain_text("") == ("", "")


# ------------------------------------------------------------------- app
def _app(monkeypatch):
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = SimpleNamespace(get=lambda k, d=None: d, user_name="Hunter")
    a._init_assistant_state()
    a._audio_busy = threading.Event()
    a._turn_busy = threading.Event()
    a._turn_timer = a._turn_watchdog = None
    a.said = []
    a._say = a.said.append
    a.context = SimpleNamespace(add_exchange=lambda u, j: a.exchanges.append((u, j)))
    a.exchanges = []
    return a


def test_async_reply_closes_the_turn_and_arms_the_follow_up(monkeypatch, cards):
    monkeypatch.setattr(CONFIG, "talkback", True)
    a = _app(monkeypatch)
    a._turn_busy.set()
    a._last_user_text, a._last_source = "explain the handout", "voice"
    a._async_reply("It's the handout, sir.")
    assert a.said == ["It's the handout, sir."]
    assert not a._turn_busy.is_set() and a._followup_after_speech
    assert a.exchanges == [("explain the handout", "It's the handout, sir.")]
    bus.drain()
    assert cards and cards[-1].text == "It's the handout, sir." and cards[-1].speak


def test_async_reply_from_typed_text_does_not_open_the_mic(monkeypatch):
    a = _app(monkeypatch)
    a._last_source = "typed"
    a._async_reply("Question 1: what is a biosensor?")
    assert not a._followup_after_speech
    time.sleep(0)
