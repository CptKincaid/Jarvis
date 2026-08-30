"""The people book (jarvis/memory.py people.json + commander "person" /
"who is"): "my advisor is Dr Peyrovi, email hp@tamu.edu" is parsed only
when the sentence is plainly about a person, stored, rendered as a People
block on every turn outside the fact window, resolved by alias / name /
address, expanded in calendar titles, and turned into a sender filter on
get_mail. Pure logic: tmp memory dirs, the substring store (semantic off),
FakeIMAP from tests/test_notes_mail.py, no network.
"""
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import jarvis.commander as commander
from jarvis.commander import ASSISTANT_TIER1, REGISTRY, Commander, IntentClassifier
from jarvis.config import CONFIG
from jarvis.memory import JarvisMemory, normalize_alias, parse_person_statement
from jarvis.tools import mail as mail_mod
from jarvis.tools.calendar import expand_people
from jarvis.tools.registry import ToolRegistry

from tests.test_notes_mail import GMAIL_CFG, FakeCfg, FakeIMAP, _canned


@pytest.fixture
def mem(tmp_path):
    return JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                        semantic=False)


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,alias,name,email", [
    ("my advisor is Dr Peyrovi, email hp@tamu.edu", "advisor", "Dr Peyrovi", "hp@tamu.edu"),
    ("My advisor is Dr. Peyrovi", "advisor", "Dr. Peyrovi", ""),
    ("my mom is Linda", "mom", "Linda", ""),
    ("my TA is Sam Ortiz, his email is sam@tamu.edu", "ta", "Sam Ortiz", "sam@tamu.edu"),
    ("my landlord is Mike Chen and his email address is mike@rent.example.",
     "landlord", "Mike Chen", "mike@rent.example"),
    ("my study partner is Priya Nair", "study partner", "Priya Nair", ""),
    ("our plumber is joe at joe@pipes.example", "plumber", "joe", "joe@pipes.example"),
])
def test_person_statements_parse(text, alias, name, email):
    got = parse_person_statement(text)
    assert got == {"alias": alias, "name": name, "email": email}


@pytest.mark.parametrize("text", [
    "my favourite colour is blue",
    "my mood is fine",
    "my thesis is due friday",
    "my password is hunter2",
    "my car is a Honda",           # capitalised single word after "a": not a name
    "remember that my dentist is",
    "what is my advisor's name",
])
def test_non_person_sentences_do_not_parse(text):
    assert parse_person_statement(text) is None


def test_normalize_alias():
    assert normalize_alias("My Advisor") == "advisor"
    assert normalize_alias("the TA!") == "ta"
    assert normalize_alias("  our  Mom ") == "mom"


# ---------------------------------------------------------------- store
def test_add_resolve_and_persist(mem, tmp_path):
    entry = mem.add_person("my advisor", "Dr Peyrovi", email="hp@tamu.edu")
    assert entry["alias"] == "advisor" and entry["relation"] == "advisor"
    again = JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "legacy",
                         semantic=False)
    assert again.resolve_person("my advisor")["name"] == "Dr Peyrovi"
    assert again.resolve_person("Advisor")["email"] == "hp@tamu.edu"
    assert again.resolve_person("peyrovi")["alias"] == "advisor"      # last name
    assert again.resolve_person("hp@tamu.edu")["alias"] == "advisor"  # address
    assert again.resolve_person("dentist") is None
    # an update keeps the address when only the name is repeated
    mem.add_person("advisor", "Dr H Peyrovi")
    assert mem.resolve_person("my advisor")["email"] == "hp@tamu.edu"
    assert mem.remove_person("my advisor") and mem.resolve_person("advisor") is None


def test_people_block_is_always_in_context_outside_the_fact_window(mem):
    mem.add_person("mom", "Linda Peyrovi")
    for i in range(8):
        mem.remember(f"f{i}", f"fact {i}")
    text = mem.format_for_context()
    assert "People (how Hunter refers to them):" in text
    assert "my mom: Linda Peyrovi" in text
    assert "fact 7" in text and "fact 0" not in text        # the last-five window
    assert "my mom" in mem.format_for_context("what's the weather")


def test_expand_aliases_whole_words_with_or_without_my(mem):
    mem.add_person("mom", "Linda Peyrovi")
    mem.add_person("advisor", "Dr Peyrovi")
    assert mem.expand_aliases("lunch with Mom") == "lunch with Linda Peyrovi"
    assert mem.expand_aliases("meeting with my advisor") == "meeting with Dr Peyrovi"
    assert mem.expand_aliases("moment of calm") == "moment of calm"   # not "mom"
    assert expand_people(SimpleNamespace(memory=mem), "coffee with the advisor") == \
        "coffee with Dr Peyrovi"
    assert expand_people(None, "coffee") == "coffee"


# ------------------------------------------------------------- commander
@pytest.fixture
def cmdr(mem, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent_log.json")
    monkeypatch.setattr(CONFIG, "voice_cmds", True)
    monkeypatch.setattr(CONFIG, "jarvis_mode", True)
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    svc = types.SimpleNamespace(desktop=MagicMock(), workflows=MagicMock(),
                                brain=MagicMock(), memory=mem, context=MagicMock(),
                                tts=MagicMock())
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.get_last_window.return_value = None
    return Commander(svc)


def test_registry_order_and_tier1_membership():
    names = [c.name for c in REGISTRY]
    assert names.index("person") < names.index("remember")
    tier1 = {c.name for c in ASSISTANT_TIER1}
    assert {"person", "remember", "recall", "who is", "recap"} <= tier1


def test_my_advisor_is_stores_a_contact_without_the_wake_word(cmdr, mem):
    res = cmdr.handle("My advisor is Dr Peyrovi, email hp@tamu.edu", source="typed")
    assert res.handled and "Dr Peyrovi" in res.reply and "hp@tamu.edu" in res.reply
    assert mem.resolve_person("my advisor")["email"] == "hp@tamu.edu"
    res = cmdr.handle("who's my advisor?", source="typed")
    assert res.handled and res.reply == "Your advisor is Dr Peyrovi, sir; hp@tamu.edu."


def test_a_non_person_my_x_is_falls_through(cmdr, mem):
    cmdr._raw_text = "my favourite colour is blue"
    assert cmdr._try_assistant("my favourite colour is blue") is None
    assert mem.people() == {}
    cmdr._raw_text = "who is my dentist"
    assert cmdr._try_assistant("who is my dentist") is None     # unknown -> the model


def test_remember_that_my_advisor_is_stores_fact_and_contact(cmdr, mem):
    res = cmdr.handle("remember that my advisor is Dr Peyrovi", source="typed")
    assert res.handled and res.reply.startswith("Remembered:")
    assert mem.recall("peyrovi")
    assert mem.resolve_person("advisor")["name"] == "Dr Peyrovi"


def test_remember_to_is_left_to_the_notes_tool(cmdr, mem):
    cmdr._raw_text = "remember to buy milk"
    assert cmdr._try_assistant("remember to buy milk") is None
    assert mem.get_all_facts() == {}


def test_recall_with_a_time_phrase(cmdr, mem, monkeypatch):
    seen = {}

    def _recall(query, **kw):
        seen.update(query=query, **kw)
        return [{"key": "k", "value": "the thesis draft is due Friday", "time": ""}]
    monkeypatch.setattr(mem, "recall", _recall)
    res = cmdr.handle("what did I say about the thesis last week", source="typed")
    assert res.handled and "thesis draft" in res.reply
    assert seen["query"] == "the thesis" and seen["since"] is not None


# ------------------------------------------------------------------ mail
@pytest.fixture
def fake_imap(monkeypatch):
    FakeIMAP.instances = []
    FakeIMAP.messages = []
    monkeypatch.setattr(FakeIMAP, "fail_login", False)
    monkeypatch.setattr(FakeIMAP, "fail_connect", False)
    return FakeIMAP


def test_get_mail_sender_filter_resolves_an_alias_to_the_address(fake_imap, mem):
    from datetime import datetime
    fake_imap.messages = _canned(datetime.now().astimezone())
    mem.add_person("advisor", "Jane Doe", email="jane@example.com")
    reg = ToolRegistry()
    reg.register_many(mail_mod.make_tools(FakeCfg(GMAIL_CFG),
                                          SimpleNamespace(imap=fake_imap, memory=mem)))
    schema = reg.schemas()[0]["function"]
    assert "sender" in schema["parameters"]["properties"]
    assert len(schema["description"].split()) <= 20
    r = reg.call("get_mail", {"sender": "my advisor"})
    assert r.ok and r.text.startswith("From Jane Doe:")
    assert "Standup moved" in r.text and "Invoice" not in r.text
    # a bare name matches the display name; an unknown alias matches literally
    r = reg.call("get_mail", {"sender": "müller"})
    assert "Invoice" in r.text and "Standup" not in r.text
    r = reg.call("get_mail", {"sender": "my dentist"})
    assert r.ok and r.speak is None and r.text == "no unread mail from my dentist in the last 24 hours"


def test_sender_matches_rules():
    m = mail_mod.Mail(from_name="Jane Doe", from_addr="jane@example.com",
                      subject="s", date=None, snippet="")
    assert mail_mod.sender_matches(m, "jane@example.com")
    assert mail_mod.sender_matches(m, "Jane")
    assert mail_mod.sender_matches(m, "example.com")
    assert not mail_mod.sender_matches(m, "bob@example.com")
    assert mail_mod.sender_matches(m, "")


def test_last_mail_handler_keeps_working(cmdr, monkeypatch):
    calls = []
    cmdr.services.brain = SimpleNamespace(chat=lambda t, **kw: calls.append(kw))
    res = cmdr.handle("what was my last email about", source="typed")
    assert res.handled and calls and calls[0]["force_tool"] == "get_mail"
    assert commander._LAST_MAIL_RX.search("my last email")
