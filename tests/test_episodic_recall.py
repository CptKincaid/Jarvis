"""Episodic recall: "when did I last talk to my advisor?" / "how long since
I worked on the thesis?" (jarvis/tools/journal.py + the commander's "last
seen" command).

The journal is a real per-day tree under tmp_path, the people book a real
JarvisMemory with semantic=False; nothing here touches Ollama, X or the
network.
"""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.commander import Commander, IntentClassifier, _LAST_SEEN_RX, _h_last_seen
from jarvis.config import CONFIG
from jarvis.memory import JarvisMemory
from jarvis.tools import journal as jm

# A Thursday, so "yesterday"/"Tuesday" wording is unambiguous.
NOW = datetime(2026, 8, 27, 15, 30)


def _write(journal_dir, when, kind, **fields):
    journal_dir.mkdir(parents=True, exist_ok=True)
    row = {"time": when.isoformat(timespec="seconds"), "kind": kind}
    row.update(fields)
    with open(journal_dir / f"{when:%Y-%m-%d}.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


@pytest.fixture
def journal(tmp_path):
    """A week of journal: the advisor on Tuesday, the thesis every day, a
    tool call and a window title as the other two row kinds."""
    d = tmp_path / "journal"
    _write(d, NOW - timedelta(days=6), "exchange",
           user="email Dr Peyrovi about the letter", jarvis="Sent, sir.")
    _write(d, NOW - timedelta(days=2, hours=2), "exchange",
           user="what did Dr Peyrovi say about the recommendation letter",
           jarvis="Nothing yet, sir.")
    _write(d, NOW - timedelta(days=1), "window",
           title="thesis_draft.tex - TeXstudio")
    _write(d, NOW - timedelta(hours=3), "tool", name="get_mail",
           args={"limit": 1}, ok=True, text="")
    _write(d, NOW - timedelta(hours=1), "exchange",
           user="quiz me on chapter three", jarvis="Five questions, sir.")
    return d


@pytest.fixture
def memory(tmp_path):
    mem = JarvisMemory(memory_dir=tmp_path / "mem", legacy_dir=tmp_path / "none",
                       semantic=False)
    mem.add_person("advisor", "Dr Peyrovi", email="hp@tamu.edu")
    return mem


# ------------------------------------------------------------ parsing
@pytest.mark.parametrize("said,target,gap", [
    ("when did i last talk to my advisor", "my advisor", False),
    ("when did i last see dr peyrovi", "dr peyrovi", False),
    ("when was the last time i worked on the thesis", "the thesis", False),
    ("when's the last time i mentioned the letter", "the letter", False),
    ("how long since i emailed my advisor", "my advisor", True),
    ("how long has it been since i opened the thesis", "the thesis", True),
    ("when did i last ask you about the recommendation letter",
     "the recommendation letter", False),
])
def test_the_matcher_finds_the_verb_and_the_thing(said, target, gap):
    m = _LAST_SEEN_RX.match(said)
    assert m is not None, said
    tail = m.group("tail1") or m.group("tail2") or m.group("tail3")
    assert jm.mention_target(tail) == target
    assert bool(m.group("gap")) is gap


def test_an_about_clause_narrows_but_does_not_widen_the_search():
    # "talked to my advisor ABOUT the letter" is a question about the
    # advisor; searching the whole phrase would match no journal row.
    assert jm.mention_target("talked to my advisor about the letter") == "my advisor"


@pytest.mark.parametrize("said", [
    "when did i last eat",              # no journal noun, but still ours
    "how long since i slept",
])
def test_bare_questions_still_match(said):
    assert _LAST_SEEN_RX.match(said) is not None


def test_neighbouring_commands_are_not_claimed():
    for other in ("what did i say about the thesis", "recap my day",
                  "how's my week looking", "when's my next exam"):
        assert _LAST_SEEN_RX.match(other) is None


# --------------------------------------------------------------- terms
def test_an_alias_expands_to_the_name_and_the_surname(memory):
    terms = [t.lower() for t in jm.mention_terms("my advisor", memory)]
    assert "my advisor" in terms and "advisor" in terms
    assert "dr peyrovi" in terms and "peyrovi" in terms


def test_terms_survive_a_memory_that_knows_nobody(tmp_path):
    mem = JarvisMemory(memory_dir=tmp_path / "m2", legacy_dir=tmp_path / "n2",
                       semantic=False)
    assert jm.mention_terms("the thesis", mem) == ["the thesis", "thesis"]


# --------------------------------------------------------------- search
def test_the_newest_mention_wins_and_the_scan_stops_there(journal):
    row = jm.find_last_mention(journal, ["Dr Peyrovi", "advisor"], now=NOW)
    assert row["kind"] == "exchange"
    assert "recommendation letter" in row["user"]
    assert row["_when"] == NOW - timedelta(days=2, hours=2)


def test_every_row_kind_is_searchable(journal):
    assert jm.find_last_mention(journal, ["TeXstudio"], now=NOW)["kind"] == "window"
    assert jm.find_last_mention(journal, ["get_mail"], now=NOW)["kind"] == "tool"


def test_a_word_inside_another_word_is_not_a_mention(journal):
    # "the sis" must not match "thesis_draft"; whole words only.
    assert jm.find_last_mention(journal, ["hesis"], now=NOW) is None
    assert jm.find_last_mention(journal, ["thesis"], now=NOW) is not None


def test_nothing_matches_and_nothing_raises(journal, tmp_path):
    assert jm.find_last_mention(journal, ["curling"], now=NOW) is None
    assert jm.find_last_mention(tmp_path / "gone", ["anything"], now=NOW) is None
    assert jm.find_last_mention(journal, [], now=NOW) is None


def test_a_row_stamped_in_the_future_is_a_clock_skew_not_a_memory(journal):
    _write(journal, NOW + timedelta(hours=2), "exchange",
           user="the thesis", jarvis="")
    row = jm.find_last_mention(journal, ["thesis"], now=NOW)
    assert row["_when"] < NOW and row["kind"] == "window"


def test_the_scan_gives_up_at_max_days(journal):
    _write(journal, NOW - timedelta(days=40), "exchange",
           user="the curling final", jarvis="")
    assert jm.find_last_mention(journal, ["curling"], now=NOW, max_days=7) is None
    assert jm.find_last_mention(journal, ["curling"], now=NOW, max_days=60) is not None


# -------------------------------------------------------------- wording
@pytest.mark.parametrize("delta,expected", [
    (timedelta(hours=2), "this afternoon"),
    (timedelta(days=1), "yesterday afternoon"),
    (timedelta(days=2), "Tuesday afternoon"),
    (timedelta(days=9), "last Tuesday"),
    (timedelta(days=40), "on the 18th of July"),
])
def test_when_words_are_spoken_words_not_a_date_stamp(delta, expected):
    assert jm.when_words(NOW - delta, NOW) == expected


def test_the_year_appears_only_when_it_is_not_this_one():
    assert "2025" in jm.when_words(NOW - timedelta(days=400), NOW)
    assert "2026" not in jm.when_words(NOW - timedelta(days=40), NOW)


@pytest.mark.parametrize("delta,expected", [
    (timedelta(minutes=30), "30 minutes"),
    (timedelta(hours=1), "about an hour"),
    (timedelta(hours=5), "about 5 hours"),
    (timedelta(days=3), "3 days"),
    (timedelta(days=14), "a fortnight"),
    (timedelta(days=90), "3 months"),
])
def test_elapsed_words(delta, expected):
    assert jm.elapsed_words(NOW - delta, NOW) == expected


def test_the_line_says_what_the_journal_saw_never_more(journal):
    row = jm.find_last_mention(journal, ["Dr Peyrovi"], now=NOW)
    line = jm.last_mention_line(row, "my advisor", now=NOW)
    assert line.startswith("Tuesday afternoon, sir.")
    # "you said" — the journal cannot know he MET her.
    assert "You said," in line and "recommendation letter" in line
    gap = jm.last_mention_line(row, "my advisor", now=NOW, duration=True)
    assert gap.startswith("It's been 2 days, sir.")


def test_a_window_row_reads_as_a_window(journal):
    row = jm.find_last_mention(journal, ["TeXstudio"], now=NOW)
    assert "open" in jm.last_mention_line(row, "the thesis", now=NOW)


# ------------------------------------------------------------- command
@pytest.fixture
def cmdr(journal, memory, tmp_path, monkeypatch):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "il.json")
    monkeypatch.setattr(CONFIG, "talkback", False)
    monkeypatch.setattr(jm, "_now", lambda: NOW)
    svc = SimpleNamespace(memory=memory, tts=MagicMock(),
                          context_engine=SimpleNamespace(
                              journal_dir=lambda: journal))
    return Commander(svc)


def _ask(cmdr, said):
    m = _LAST_SEEN_RX.match(said)
    assert m is not None, said
    return _h_last_seen(cmdr, said, m)


def test_the_command_answers_from_the_journal_through_the_people_book(cmdr):
    res = _ask(cmdr, "when did i last talk to my advisor")
    assert res.handled and res.speak
    assert res.reply.startswith("Tuesday afternoon, sir.")
    assert "recommendation letter" in res.reply


def test_how_long_since_gets_a_duration(cmdr):
    res = _ask(cmdr, "how long since i worked on the thesis")
    assert res.reply.startswith("It's been a day, sir.")


def test_a_journal_miss_falls_back_to_what_he_told_me(cmdr, memory):
    memory.remember("dentist", "my dentist is Dr Patel")
    res = _ask(cmdr, "when did i last see my dentist")
    assert "Nothing in the journal" in res.reply
    assert "Dr Patel" in res.reply


def test_a_miss_with_nothing_stored_says_so_plainly(cmdr):
    res = _ask(cmdr, "when did i last go curling")
    assert res.reply == "Nothing in the journal about curling, sir."
    assert res.handled and res.speak


def test_an_empty_target_falls_through_to_the_model(cmdr):
    m = _LAST_SEEN_RX.match("when did i last see")
    assert _h_last_seen(cmdr, "when did i last see", m) is None


def test_a_stubbed_memory_never_reaches_the_voice(journal, tmp_path, monkeypatch):
    """recall() on a MagicMock returns a MagicMock, which f-strings into
    "<MagicMock id=...>" — the guard is why the miss line stays English."""
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "il2.json")
    monkeypatch.setattr(CONFIG, "talkback", False)
    svc = SimpleNamespace(memory=MagicMock(), tts=MagicMock(),
                          context_engine=SimpleNamespace(journal_dir=lambda: journal))
    c = Commander(svc)
    res = _ask(c, "when did i last go curling")
    assert "MagicMock" not in (res.reply or "")


def test_an_unreadable_journal_is_an_excuse_not_a_traceback(cmdr, monkeypatch):
    monkeypatch.setattr(jm, "find_last_mention",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    res = _ask(cmdr, "when did i last talk to my advisor")
    assert res.handled and "couldn't read the journal" in res.reply


def test_the_registry_routes_it_and_recall_still_owns_its_own_words(cmdr):
    res = cmdr.handle("jarvis when did i last talk to my advisor", source="typed")
    assert res.handled and "Tuesday afternoon" in (res.reply or "")
    assert cmdr._match_assistant("when did i last talk to my advisor") == "last seen"
    assert cmdr._match_assistant("what did i say about the thesis") == "recall"
