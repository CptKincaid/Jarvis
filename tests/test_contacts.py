"""The address book (jarvis/contacts.py, scripts/jarvis_contacts.py) and
what it does to the send lane.

Sending a document to the wrong person cannot be undone, so most of this
file is about the ways the book must REFUSE or ASK: a misheard name
resolves to nobody, two Heathers are a question and never the first one,
a malformed address never gets into the file, and nothing spoken writes
it. Every name and address here is made up; the book under test lives in
tmp_path through JARVIS_CONTACTS, never in ~/.config.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import stat
from pathlib import Path

import pytest

from jarvis import contacts as book_mod
from jarvis import outbox
from tests.test_send_file import (Cfg, FakeSMTP, cfg_with_roots,  # noqa: F401
                                  cmd, home, roots, _fresh_fakes)  # - fixtures

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jarvis_contacts.py"

TWO_HEATHERS = [
    {"name": "Heather Smith", "email": "heather@example.com",
     "honorific": "Dr", "aliases": ["my advisor"], "note": "PhD advisor"},
    {"name": "Heather Jones", "email": "hjones@example.com"},
    {"name": "Mum", "email": "linda@example.com", "aliases": ["mom", "my mother"]},
]


# ------------------------------------------------------------- fixtures
@pytest.fixture
def book_file(tmp_path, monkeypatch):
    """The book lives here and nowhere else for the test."""
    path = tmp_path / "cfg" / "contacts.json"
    monkeypatch.setenv(book_mod.ENV_VAR, str(path))
    monkeypatch.setattr(book_mod, "_BOOK", None)
    yield path
    monkeypatch.setattr(book_mod, "_BOOK", None)


def write(path: Path, rows, **top):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": 1, "contacts": rows}
    payload.update(top)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


@pytest.fixture
def two_heathers(book_file):
    write(book_file, TWO_HEATHERS)
    return book_file


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("jarvis_contacts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==================================================================
# 1. Entry validation: a typo cannot sit there waiting to be mailed to
# ==================================================================
@pytest.mark.parametrize("bad", [
    "heather", "heather@", "@example.com", "heather@example",
    "heather @example.com", "heather@ example.com", "heather@example.com junk",
    "heather@@example.com", "heather@example..com", "", "   ",
    "heather at example dot com", "<heather@example.com>",
])
def test_a_malformed_address_is_refused_on_entry(bad):
    contact, why = book_mod.validate_row({"name": "Heather Smith", "email": bad})
    assert contact is None
    assert "address" in why or "missing" in why, why


def test_the_entry_check_is_a_full_match_not_a_search():
    """outbox.parse_address SEARCHES text and would take the address out of
    "heather@example.com junk"; the book must not."""
    assert outbox.parse_address("heather@example.com junk") == "heather@example.com"
    contact, _ = book_mod.validate_row({"name": "H", "email": "heather@example.com junk"})
    assert contact is None


@pytest.mark.parametrize("bad", ["heather@example.com", "Heather <h@x.com>",
                                 "", "   ", "H" * 81, "Heather; Smith",
                                 "12345", "Heather_Smith", "Heather/Smith"])
def test_a_bad_name_is_refused(bad):
    contact, why = book_mod.validate_row({"name": bad, "email": "h@example.com"})
    assert contact is None, (bad, why)


def test_a_good_row_validates_and_keeps_what_it_does_not_know():
    contact, why = book_mod.validate_row(
        {"name": "  Heather   Smith ", "email": " Heather@Example.com ",
         "honorific": "Dr", "aliases": ["my advisor", "my advisor", ""],
         "note": "PhD advisor", "phone": "555-0100"})
    assert why == "" and contact is not None
    assert contact.name == "Heather Smith"
    assert contact.email == "Heather@Example.com"
    assert contact.aliases == ["my advisor"]
    assert contact.extra == {"phone": "555-0100"}
    assert contact.row()["phone"] == "555-0100"
    assert contact.spoken == "Dr Heather Smith"
    assert contact.first == "heather" and contact.surname == "smith"


def test_a_one_word_name_is_both_first_and_surname():
    contact, _ = book_mod.validate_row({"name": "Mum", "email": "m@example.com"})
    assert contact.first == contact.surname == "mum"


@pytest.mark.parametrize("row, why_has", [
    ({"name": "heather smith", "email": "other@example.com"}, "already in the book"),
    ({"name": "Someone Else", "email": "HEATHER@example.com"}, "already in the book"),
    ({"name": "Someone Else", "email": "x@example.com", "aliases": ["heather"]},
     "already how"),
    ({"name": "Someone Else", "email": "x@example.com", "aliases": ["Smith"]},
     "already how"),
    ({"name": "Someone Else", "email": "x@example.com", "aliases": ["the advisor"]},
     "already how"),
    ({"name": "Advisor", "email": "x@example.com"}, "already an alias"),
])
def test_the_uniqueness_rules(row, why_has):
    existing, _ = book_mod.validate_row(TWO_HEATHERS[0])
    new, why = book_mod.validate_row(row)
    assert new is not None, why
    assert why_has in book_mod.check_unique([existing], new)


# ==================================================================
# 2. The file: his, 0600, atomic, hand-editable, hot
# ==================================================================
def test_add_creates_the_file_0600_in_a_0700_dir_and_leaves_no_temp(book_file):
    assert not book_file.exists()
    book = book_mod.current()
    contact, why = book.add({"name": "Heather Smith", "email": "heather@example.com"})
    assert why == "" and contact is not None and contact.index == 0
    assert stat.S_IMODE(book_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(book_file.parent.stat().st_mode) == 0o700
    assert [p.name for p in book_file.parent.iterdir()] == ["contacts.json"], \
        "the temp file of the atomic write must be gone"
    data = json.loads(book_file.read_text())
    assert data["format"] == 1
    assert data["contacts"] == [{"name": "Heather Smith",
                                 "email": "heather@example.com"}]


def test_the_write_is_atomic_through_a_temp_file_in_the_same_dir(book_file, monkeypatch):
    """The dance: mkstemp beside the file, fsync, os.replace. A crash in
    the middle leaves the old book, never half of a new one."""
    write(book_file, TWO_HEATHERS[:1])
    before = book_file.read_text()
    seen = []
    real_replace = os.replace

    def _replace(src, dst):
        seen.append((Path(src).parent, Path(src).name))
        raise OSError("disk went away")

    monkeypatch.setattr(book_mod.os, "replace", _replace)
    with pytest.raises(OSError):
        book_mod.current().add({"name": "Heather Jones", "email": "hj@example.com"})
    monkeypatch.setattr(book_mod.os, "replace", real_replace)
    assert book_file.read_text() == before, "the old book survives a failed write"
    assert seen and seen[0][0] == book_file.parent
    assert seen[0][1].startswith(".contacts-")
    assert [p.name for p in book_file.parent.iterdir()] == ["contacts.json"]


def test_a_refused_add_writes_nothing(book_file):
    write(book_file, TWO_HEATHERS[:1])
    stamp = book_file.stat().st_mtime_ns
    book = book_mod.current()
    for row in ({"name": "Heather Smith", "email": "dup@example.com"},
                {"name": "Dana Ruiz", "email": "not an address"},
                {"name": "Dana Ruiz", "email": "heather@example.com"}):
        contact, why = book.add(row)
        assert contact is None and why
    assert book_file.stat().st_mtime_ns == stamp
    assert len(json.loads(book_file.read_text())["contacts"]) == 1


def test_unknown_keys_survive_a_rewrite(book_file):
    write(book_file, [dict(TWO_HEATHERS[0], phone="555-0100")], comment="mine")
    book = book_mod.current()
    book.add({"name": "Heather Jones", "email": "hj@example.com"})
    data = json.loads(book_file.read_text())
    assert data["comment"] == "mine"
    assert data["contacts"][0]["phone"] == "555-0100"
    assert data["contacts"][1]["name"] == "Heather Jones"


def test_a_bad_row_is_skipped_flagged_and_kept_on_rewrite(book_file, caplog):
    write(book_file, [{"name": "Dana Ruiz", "email": "dana@example"},
                      TWO_HEATHERS[0]])
    with caplog.at_level(logging.WARNING, logger="contacts"):
        book = book_mod.current()
    assert [c.name for c in book.contacts] == ["Heather Smith"]
    assert len(book.skipped) == 1
    assert book.skipped[0].index == 0 and book.skipped[0].name == "Dana Ruiz"
    assert "bad address" in book.skipped[0].why
    assert book.resolve("Dana").unknown, "a flagged row is never resolved"
    # the loader names the row and never the address
    assert "Dana Ruiz" in caplog.text and "dana@example" not in caplog.text
    # the bad row is his to fix: a page write does not throw it away
    book.add({"name": "Heather Jones", "email": "hj@example.com"})
    data = json.loads(book_file.read_text())
    assert data["contacts"][0] == {"name": "Dana Ruiz", "email": "dana@example"}


def test_a_duplicate_on_load_keeps_the_first_and_flags_the_second(book_file):
    write(book_file, [TWO_HEATHERS[0],
                      {"name": "heather smith", "email": "other@example.com"}])
    book = book_mod.current()
    assert [c.email for c in book.contacts] == ["heather@example.com"]
    assert book.skipped and book.skipped[0].index == 1


def test_a_missing_file_is_an_empty_book_not_an_error(book_file, caplog):
    with caplog.at_level(logging.DEBUG, logger="contacts"):
        book = book_mod.current()
    assert book.contacts == [] and book.skipped == []
    assert book.resolve("Heather").unknown
    assert caplog.text == ""


def test_a_corrupt_edit_keeps_the_last_good_book_and_logs_the_path_only(
        two_heathers, caplog):
    book = book_mod.current()
    assert book.resolve("Heather Smith").found
    two_heathers.write_text("{ this is not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="contacts"):
        assert book_mod.current().resolve("Heather Smith").found
        assert book_mod.current().resolve("Heather Smith").found
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "one warning per stamp, not one per resolve"
    assert str(two_heathers) in warnings[0].getMessage()
    assert "example.com" not in caplog.text


@pytest.mark.parametrize("text", ['{"format": 2, "contacts": []}',
                                  '{"contacts": "nope", "format": 1}',
                                  '[]'])
def test_a_wrong_format_is_corrupt_not_empty(two_heathers, text):
    assert book_mod.current().resolve("Heather Smith").found
    two_heathers.write_text(text, encoding="utf-8")
    assert book_mod.current().resolve("Heather Smith").found


def test_an_edit_by_hand_is_seen_on_the_next_resolve_without_a_restart(book_file):
    write(book_file, [{"name": "Heather Smith", "email": "heather@example.com"}])
    book = book_mod.current()
    assert book.resolve("Heather").addr == "heather@example.com"
    # his editor writes a new row; nothing is restarted, nothing is told
    write(book_file, [{"name": "Heather Smith", "email": "heather@example.com"},
                      {"name": "Dana Ruiz", "email": "dana@example.com"}])
    assert book_mod.current().resolve("Dana").addr == "dana@example.com"
    # and a row taken away is gone at once
    write(book_file, [{"name": "Dana Ruiz", "email": "dana@example.com"}])
    assert book_mod.current().resolve("Heather").unknown


def test_the_file_is_not_re_read_when_the_stamp_has_not_moved(two_heathers, monkeypatch):
    book = book_mod.current()
    calls = []
    real = Path.read_text

    def _read(self, *a, **k):
        calls.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _read)
    for _ in range(5):
        book.resolve("Heather Smith")
    assert calls == [], "one stat per resolve, and no read while it is unchanged"


def test_the_book_is_its_own_file_beside_the_config(monkeypatch, tmp_path):
    monkeypatch.delenv(book_mod.ENV_VAR, raising=False)
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert book_mod.book_path() == tmp_path / "contacts.json"
    assert book_mod.book_path().name != "people.json"


# ==================================================================
# 3. Resolution: exact, or a question, or nothing
# ==================================================================
def test_an_ambiguous_first_name_asks_and_never_picks(two_heathers):
    res = book_mod.current().resolve("Heather")
    assert res.ambiguous and not res.found
    assert res.candidates == ["Heather Smith", "Heather Jones"], "file order"
    assert res.addr == ""


def test_an_exact_full_name_wins_outright(two_heathers):
    for said in ("Heather Smith", "heather smith", "HEATHER SMITH.", "Dr Heather Smith",
                 "Dr. Heather Smith"):
        res = book_mod.current().resolve(said)
        assert res.found and res.addr == "heather@example.com", said
        assert res.name == "Heather Smith" and res.from_book
        assert res.honorific == "Dr"


@pytest.mark.parametrize("said", ["Heathr", "Heath", "Heather S", "Heather Smyth",
                                  "Heather Smith Jones", "Smithy", "advisors",
                                  "Dr", "Dr Heather", "", "   ", "@"])
def test_a_misheard_name_resolves_to_nobody(two_heathers, said):
    res = book_mod.current().resolve(said)
    assert res.unknown, (said, res)


@pytest.mark.parametrize("said, want", [
    ("Smith", "heather@example.com"), ("Jones", "hjones@example.com"),
    ("Dr Smith", "heather@example.com"), ("dr. smith", "heather@example.com"),
    ("my advisor", "heather@example.com"), ("advisor", "heather@example.com"),
    ("the advisor", "heather@example.com"),
    ("Mum", "linda@example.com"), ("mom", "linda@example.com"),
    ("my mother", "linda@example.com"), ("mother", "linda@example.com"),
])
def test_surname_honorific_and_alias_are_exact_keys(two_heathers, said, want):
    res = book_mod.current().resolve(said)
    assert res.found and res.addr == want, (said, res)


def test_a_shared_surname_is_a_question_too(book_file):
    write(book_file, [{"name": "Heather Smith", "email": "h@example.com"},
                      {"name": "John Smith", "email": "j@example.com"}])
    res = book_mod.current().resolve("Smith")
    assert res.ambiguous and res.candidates == ["Heather Smith", "John Smith"]
    assert book_mod.current().resolve("Heather").addr == "h@example.com"


def test_the_which_answer_takes_only_an_exact_name(two_heathers):
    book = book_mod.current()
    cands = ["Heather Smith", "Heather Jones"]
    assert book.choose("Heather Smith", cands) == "Heather Smith"
    assert book.choose("smith", cands) == "Heather Smith"
    assert book.choose("Dr Smith", cands) == "Heather Smith"
    assert book.choose("Dr. Heather Smith", cands) == "Heather Smith"
    assert book.choose("Jones", cands) == "Heather Jones"
    # the first name is the ambiguity being asked about
    assert book.choose("Heather", cands) is None
    assert book.choose("Heather Smit", cands) is None
    assert book.choose("Smiths", cands) is None
    assert book.choose("Mum", cands) is None, "not one of the rivals"


def test_the_which_answer_refuses_a_surname_the_rivals_share(book_file):
    write(book_file, [{"name": "Heather Smith", "email": "h@example.com"},
                      {"name": "Heather Smith-Jones", "email": "j@example.com"},
                      {"name": "John Smith", "email": "js@example.com"}])
    book = book_mod.current()
    assert book.resolve("Smith").ambiguous
    assert book.choose("Smith", ["Heather Smith", "John Smith"]) is None
    assert book.choose("John Smith", ["Heather Smith", "John Smith"]) == "John Smith"


def test_the_which_question_names_the_rivals_in_file_order():
    assert book_mod.which_line("Heather", ["Heather Smith", "Heather Jones"]) == \
        "Which Heather, sir — Heather Smith or Heather Jones?"
    assert book_mod.which_line("heather.", ["Heather Smith", "Heather Jones",
                                            "Heather Brown"]) == \
        "Which heather, sir — Heather Smith, Heather Jones or Heather Brown?"
    five = [f"Heather {s}" for s in ("A", "B", "C", "D", "E")]
    assert book_mod.which_line("Heather", five) == \
        "I've five people called Heather, sir — the full name, please."
    assert book_mod.reask_line(["Heather Smith", "Heather Jones"]) == \
        "The full name, sir — Heather Smith or Heather Jones?"


def test_no_scorer_is_anywhere_near_the_book():
    """Invariant 1: string equality only. filepick's scorer and the file
    pick's fuzzy answer must not be importable from the book's code."""
    src = Path(book_mod.__file__).read_text()
    for word in ("filepick", "score(", "SequenceMatcher", "difflib",
                 "levenshtein", "startswith(", "pick_from_answer"):
        assert word not in src, word


# ==================================================================
# 4. outbox: the book comes first, and the read-back speaks the NAME
# ==================================================================
def test_the_book_is_consulted_before_the_legacy_map(two_heathers):
    cfg = Cfg(**{"send_file.contacts": {"heather smith": "legacy@example.com",
                                        "dana": "dana@example.com"}})
    res = outbox.resolve(cfg, None, "Heather Smith")
    assert res.addr == "heather@example.com" and res.from_book
    assert outbox.resolve_recipient(cfg, None, "Heather Smith") == \
        ("heather@example.com", "Heather Smith")
    # and the legacy map still answers when the book is silent
    res = outbox.resolve(cfg, None, "Dana")
    assert res.addr == "dana@example.com" and not res.from_book


def test_an_ambiguous_book_name_beats_a_legacy_row_that_would_have_guessed(two_heathers):
    cfg = Cfg(**{"send_file.contacts": {"heather": "legacy@example.com"}})
    res = outbox.resolve(cfg, None, "Heather")
    assert res.ambiguous and res.addr == ""
    assert outbox.resolve_recipient(cfg, None, "Heather") == ("", "")


def test_memory_still_answers_my_brother_with_no_book_row(two_heathers):
    import types
    memory = types.SimpleNamespace(
        resolve_person=lambda t: {"name": "Sam Peyrovi", "email": "sam@example.com"}
        if "brother" in t.lower() else None)
    res = outbox.resolve(Cfg(**{"send_file.contacts": {}}), memory, "my brother")
    assert res.addr == "sam@example.com" and res.name == "Sam Peyrovi"
    assert not res.from_book


def test_prepare_asks_which_person_with_its_own_status(two_heathers, roots):  # noqa: F811
    cfg = cfg_with_roots(roots, **{"send_file.from": "school", "send_file.contacts": {}})
    prep = outbox.prepare(cfg, None, "the biosensors handout", "Heather")
    assert prep.draft is None
    assert prep.status == outbox.WHICH_PERSON_STATUS == "Which person?"
    assert prep.ask == "Which Heather, sir — Heather Smith or Heather Jones?"
    assert prep.candidates == ["Heather Smith", "Heather Jones"]


def test_the_read_back_speaks_the_name_and_never_the_address(two_heathers, roots):  # noqa: F811
    cfg = cfg_with_roots(roots, **{"send_file.from": "school", "send_file.contacts": {}})
    prep = outbox.prepare(cfg, None, "the biosensors handout", "Jones")
    assert prep.ok and prep.draft.from_book
    line = outbox.read_back(prep.draft)
    assert line == ("Biosensors Lab Handout v2.pdf, 5 kilobytes, to Heather Jones, "
                    "from your school account. Send it, sir?")
    assert "example" not in line and "@" not in line and " dot " not in line
    assert outbox.shown_address(prep.draft) == "to hjones@example.com"
    unsure = outbox.unsure_line(prep.draft)
    assert "to Heather Jones." in unsure and "example" not in unsure


def test_the_honorific_is_spoken_before_the_name(two_heathers, roots):  # noqa: F811
    cfg = cfg_with_roots(roots, **{"send_file.from": "school", "send_file.contacts": {}})
    prep = outbox.prepare(cfg, None, "the biosensors handout", "my advisor")
    assert prep.ok
    assert ", to Dr Heather Smith, from" in outbox.read_back(prep.draft)
    assert outbox.shown_address(prep.draft) == "to heather@example.com"


def test_a_spoken_address_and_a_legacy_hit_keep_the_spelled_out_wording(two_heathers, roots):  # noqa: F811
    """Deliberately unchanged: they have no validated name to say instead."""
    cfg = cfg_with_roots(roots, **{"send_file.from": "school",
                                   "send_file.contacts": {"dana": "dana@example.com"}})
    prep = outbox.prepare(cfg, None, "the biosensors handout", "dana@example.com")
    assert "to dana at example dot com" in outbox.read_back(prep.draft)
    assert outbox.shown_address(prep.draft) == ""
    prep = outbox.prepare(cfg, None, "the biosensors handout", "Dana")
    assert "to Dana, at dana at example dot com" in outbox.read_back(prep.draft)
    assert outbox.shown_address(prep.draft) == ""


# ==================================================================
# 5. The commander: "Which Heather, sir?" is a question he can answer
# ==================================================================
@pytest.fixture
def cmd_book(cmd, two_heathers):  # noqa: F811
    cmd.services.assistant.data["send_file.contacts"] = {}
    return cmd


def test_two_heathers_ask_and_arm_nothing(cmd_book):
    res = cmd_book.handle("email the biosensors handout to Heather", source="voice")
    assert res.reply == "Which Heather, sir — Heather Smith or Heather Jones?"
    assert res.status == "Which person?"
    assert cmd_book._pending_send is None and cmd_book._pending_filepick is None
    assert cmd_book._pending_sendask is not None
    assert cmd_book._pending_sendask.kind == "person"
    assert cmd_book._pending_sendask.candidates == ["Heather Smith", "Heather Jones"]
    assert cmd_book.question_open()
    assert not FakeSMTP.made


def test_shape_b_with_two_heathers_reaches_the_question_not_silence(cmd_book):
    res = cmd_book.handle("send Heather the biosensors handout", source="voice")
    assert res is not None and res.reply.startswith("Which Heather, sir")


@pytest.mark.parametrize("said, who, addr", [
    ("Heather Smith", "Dr Heather Smith", "heather@example.com"),
    ("Smith", "Dr Heather Smith", "heather@example.com"),
    ("Dr Smith", "Dr Heather Smith", "heather@example.com"),
    ("Dr. Heather Smith, please", "Dr Heather Smith", "heather@example.com"),
    ("the first one", "Dr Heather Smith", "heather@example.com"),
    ("Jones", "Heather Jones", "hjones@example.com"),
    ("Heather Jones", "Heather Jones", "hjones@example.com"),
    ("the second one", "Heather Jones", "hjones@example.com"),
    ("2", "Heather Jones", "hjones@example.com"),
    ("the other one", "Heather Jones", "hjones@example.com"),
    ("the last one", "Heather Jones", "hjones@example.com"),
    ("to Jones", "Heather Jones", "hjones@example.com"),
])
def test_the_which_answer_reads_back_that_person_by_name(cmd_book, said, who, addr):
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    res = cmd_book.handle(said, source="voice")
    assert res is not None and res.reply.endswith("Send it, sir?"), (said, res)
    assert f"to {who}, from your school account" in res.reply
    assert "example" not in res.reply, "the address is never spoken"
    assert res.display_only == f"to {addr}", "and it is shown"
    assert cmd_book._pending_send is not None and cmd_book._pending_sendask is None
    assert not FakeSMTP.made


def test_the_yes_then_sends_to_the_book_address_and_says_the_name(cmd_book):
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    cmd_book.handle("Jones", source="voice")
    res = cmd_book.handle("yes", source="voice")
    assert res.ack
    assert FakeSMTP.made[-1].sent[0]["To"] == "hjones@example.com"
    assert cmd_book.spoken == ["Sent to Heather Jones, sir."]


def test_the_first_name_again_is_asked_once_more_then_let_go(cmd_book):
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    res = cmd_book.handle("Heather", source="voice")
    assert res.reply == "The full name, sir — Heather Smith or Heather Jones?"
    assert cmd_book.question_open()
    res = cmd_book.handle("Heather", source="voice")
    assert res.reply == outbox.ASK_DROPPED_LINE
    assert cmd_book._pending_sendask is None and not FakeSMTP.made


@pytest.mark.parametrize("said", ["yes", "Heathr", "Heather Smyth", "Mum"])
def test_a_near_miss_or_a_bare_yes_is_never_a_pick(cmd_book, said):
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    res = cmd_book.handle(said, source="voice")
    assert res.reply.startswith("The full name, sir"), (said, res)
    assert cmd_book._pending_send is None and not FakeSMTP.made


def test_a_no_lets_it_go_out_loud(cmd_book):
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    res = cmd_book.handle("never mind", source="voice")
    assert res.reply == outbox.ASK_SPENT_LINE
    assert cmd_book._pending_sendask is None and not FakeSMTP.made


def test_a_new_subject_drops_the_question_and_keeps_its_meaning(cmd_book):
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    res = cmd_book.handle("what is the capital of france and why", source="voice")
    assert res is None or "Which" not in (res.reply or "")
    assert cmd_book._pending_sendask is None and not FakeSMTP.made


def test_the_question_is_not_answered_from_another_room(cmd_book):
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    cmd_book.handle("Heather Smith", source="phone")
    assert cmd_book._pending_sendask is not None, "parked, not spent"
    assert cmd_book._pending_send is None
    res = cmd_book.handle("Heather Smith", source="voice")
    assert "to Dr Heather Smith" in res.reply


def test_an_expired_question_ignores_a_late_answer(cmd_book, monkeypatch):
    from jarvis import commander as cm
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    ask = cmd_book._pending_sendask
    monkeypatch.setattr(cm, "SENDASK_TTL_S", 0.0)
    ask.made_at -= 1.0
    res = cmd_book.handle("Heather Smith", source="voice")
    assert cmd_book._pending_send is None
    assert res is None or "Send it, sir?" not in (res.reply or "")


def test_the_address_question_answered_with_an_ambiguous_name_asks_which(cmd_book):
    res = cmd_book.handle("email the biosensors handout to Dana", source="voice")
    assert res.reply == "I've no address for Dana, sir. What is it?"
    res = cmd_book.handle("send it to Heather instead", source="voice")
    assert res.reply == "Which Heather, sir — Heather Smith or Heather Jones?"
    assert cmd_book._pending_sendask.kind == "person"
    res = cmd_book.handle("Smith", source="voice")
    assert "to Dr Heather Smith" in res.reply and not FakeSMTP.made


def test_a_correction_on_the_read_back_goes_through_the_book(cmd_book):
    cmd_book.handle("email the biosensors handout to Heather Smith", source="voice")
    res = cmd_book.handle("yes, send it to Jones instead", source="voice")
    assert "to Heather Jones, from" in res.reply and "Smith" not in res.reply
    assert res.display_only == "to hjones@example.com"
    assert not FakeSMTP.made
    cmd_book.handle("yes", source="voice")
    assert FakeSMTP.made[-1].sent[0]["To"] == "hjones@example.com"


def test_a_correction_to_an_ambiguous_name_asks_which(cmd_book):
    """A correction rides on a yes ("yes, but to Heather" -- F23); a bare
    "no, to Heather" is a no and drops the draft, exactly as today."""
    cmd_book.handle("email the biosensors handout to Mum", source="voice")
    res = cmd_book.handle("yes, but to Heather", source="voice")
    assert res.reply == "Which Heather, sir — Heather Smith or Heather Jones?"
    assert cmd_book._pending_send is None and not FakeSMTP.made


def test_a_pronoun_yes_still_sends_to_the_book_person(cmd_book):
    cmd_book.handle("email the biosensors handout to Mum", source="voice")
    cmd_book.handle("yes, send it to her", source="voice")
    assert FakeSMTP.made[-1].sent[0]["To"] == "linda@example.com"
    assert cmd_book.spoken == ["Sent to Mum, sir."]


def test_the_vague_re_ask_names_the_person_not_the_address(cmd_book):
    cmd_book.handle("email the biosensors handout to Mum", source="voice")
    res = cmd_book.handle("okay", source="voice")
    assert res.reply == ("I'd rather be certain, sir — that's Biosensors Lab Handout "
                         "v2.pdf to Mum. Say yes and I'll send it, or no and I'll let "
                         "it go.")


def test_an_edit_to_the_file_is_seen_by_the_next_send_without_a_restart(cmd_book,
                                                                       two_heathers):
    res = cmd_book.handle("email the biosensors handout to Dana", source="voice")
    assert res.reply == "I've no address for Dana, sir. What is it?"
    cmd_book.handle("never mind", source="voice")
    write(two_heathers, TWO_HEATHERS + [{"name": "Dana Ruiz",
                                         "email": "dana@example.com"}])
    res = cmd_book.handle("email the biosensors handout to Dana", source="voice")
    assert "to Dana Ruiz, from your school account" in res.reply
    assert res.display_only == "to dana@example.com"


def test_nothing_spoken_writes_the_book(cmd_book, two_heathers):
    before = two_heathers.read_text()
    stamp = two_heathers.stat().st_mtime_ns
    cmd_book.handle("email the biosensors handout to Dana", source="voice")
    cmd_book.handle("dana at example dot com", source="voice")
    cmd_book.handle("yes", source="voice")
    assert FakeSMTP.made[-1].sent[0]["To"] == "dana@example.com"
    cmd_book.handle("remember that Dana's address is dana at example dot com",
                    source="voice")
    assert two_heathers.read_text() == before
    assert two_heathers.stat().st_mtime_ns == stamp
    assert book_mod.current().resolve("Dana").unknown


def test_the_person_question_never_reaches_the_file_offer(cmd_book, monkeypatch):
    """Invariant 1: the file offer's answer is SCORED. A list of people must
    branch before it, and no test above may have passed by accident."""
    from jarvis import commander as cm
    monkeypatch.setattr(cm, "_send_file_offer",
                        lambda *a, **k: pytest.fail("people reached the file offer"))
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    assert cmd_book._pending_sendask.kind == "person"
    monkeypatch.setattr(cm, "pick_from_answer",
                        lambda *a, **k: pytest.fail("a name was scored"))
    res = cmd_book.handle("Heather Smyth", source="voice")
    assert res.reply.startswith("The full name, sir")


# ---- the gender seam (the confirm-shapes merge, 09-05) -------------------
# Hunter's 19:00 ruling: a pronoun in a yes has to be the pending person's,
# and the gender comes from an EXPLICIT source only. The book row's stored
# honorific is that source for a book-resolved draft: Draft.to_gender is
# filled from it (outbox.draft_gender), "Dr" and a bare row leave it None.
TITLED = [
    {"name": "Dana Ruiz", "email": "dana@example.com", "honorific": "Mrs"},
    {"name": "Sam Ortiz", "email": "sam@example.com", "honorific": "Mr"},
    {"name": "Heather Smith", "email": "heather@example.com", "honorific": "Dr"},
    {"name": "Heather Jones", "email": "hjones@example.com", "honorific": "Ms"},
]


@pytest.fixture
def cmd_titled(cmd, book_file):  # noqa: F811
    write(book_file, TITLED)
    cmd.services.assistant.data["send_file.contacts"] = {}
    return cmd


@pytest.mark.parametrize("said, gender, wrong, right, who, addr", [
    ("Dana", "f", "send it to him", "send it to her", "Dana Ruiz", "dana@example.com"),
    ("Mrs Ruiz", "f", "yes, to him", "yes, to her", "Dana Ruiz", "dana@example.com"),
    ("Sam", "m", "send it to her", "send it to him", "Sam Ortiz", "sam@example.com"),
    ("Ortiz", "m", "okay send it to her", "okay send it to him", "Sam Ortiz",
     "sam@example.com"),
])
def test_a_book_rows_honorific_fills_the_gender_seam(cmd_titled, said, gender,
                                                     wrong, right, who, addr):
    """The wrong-gender pronoun re-asks and keeps the draft; the right one
    sends, to the book address, and nothing before it."""
    c = cmd_titled
    res = c.handle(f"email the biosensors handout to {said}", source="voice")
    assert res.reply.endswith("Send it, sir?"), res.reply
    assert c._pending_send is not None
    assert c._pending_send.from_book and c._pending_send.to_gender == gender
    res = c.handle(wrong, source="voice")
    assert not FakeSMTP.made, (said, wrong, res)
    assert res is not None and res.handled and res.speak
    assert c._pending_send is not None, "the draft is kept for the re-ask"
    assert res.reply == outbox.gender_line(c._pending_send)
    assert res.reply == f"The draft is to {who}, sir. Send it to {outbox.pronoun_for(gender)}?"
    res = c.handle(right, source="voice")
    assert res.ack, (said, right, res)
    assert len(FakeSMTP.made) == 1 and FakeSMTP.made[-1].sent[0]["To"] == addr
    assert c._pending_send is None


@pytest.mark.parametrize("pronoun", ["send it to him", "send it to her"])
def test_a_dr_row_leaves_the_gender_unknown_and_either_pronoun_sends(cmd_titled, pronoun):
    c = cmd_titled
    c.handle("email the biosensors handout to Heather Smith", source="voice")
    assert c._pending_send is not None and c._pending_send.from_book
    assert c._pending_send.to_gender is None, "Dr says nothing; the name is never read"
    res = c.handle(pronoun, source="voice")
    assert res.ack and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_the_which_answer_carries_the_picked_rows_honorific(cmd_titled):
    """The row he picks by answering "Which Heather?" goes back through
    prepare with the recipient settled, and its honorific fills the seam
    the same way: Ms Heather Jones is "her"."""
    c = cmd_titled
    res = c.handle("email the biosensors handout to Heather", source="voice")
    assert res.reply == "Which Heather, sir — Heather Smith or Heather Jones?"
    c.handle("Jones", source="voice")
    assert c._pending_send is not None and c._pending_send.to_gender == "f"
    res = c.handle("send it to him", source="voice")
    assert not FakeSMTP.made and res.reply == "The draft is to Heather Jones, sir. Send it to her?"
    res = c.handle("yes, send it to her", source="voice")
    assert res.ack and FakeSMTP.made[-1].sent[0]["To"] == "hjones@example.com"


def test_draft_gender_reads_the_row_first_then_what_he_said(roots):  # noqa: F811
    cfg = cfg_with_roots(roots, **{"send_file.contacts": {"mr jones": "jones@example.com"}})
    book_row = book_mod.Resolution(addr="dana@example.com", name="Dana Ruiz",
                                   from_book=True, honorific="Mrs")
    assert outbox.draft_gender(cfg, None, "Dana", book_row) == "f"
    dr_row = book_mod.Resolution(addr="h@example.com", name="Heather Smith",
                                 from_book=True, honorific="Dr")
    assert outbox.draft_gender(cfg, None, "Heather", dr_row) is None
    assert outbox.draft_gender(cfg, None, "Mrs Smith", dr_row) == "f", "what he said still counts"
    legacy = book_mod.Resolution(addr="jones@example.com", name="Jones")
    assert outbox.draft_gender(cfg, None, "Jones", legacy) == "m", "the legacy key's Mr"
    assert outbox.draft_gender(cfg, None, "Dana", book_mod.Resolution()) is None


# ==================================================================
# 6. The CLI: file-direct, works with Jarvis down, and `show` is the
#    numbers-only instrument for the book
# ==================================================================
def test_the_cli_never_reads_the_secrets_file():
    """File-direct: the script imports jarvis.contacts and nothing that
    could open assistant.json or a socket. The docstring may say the
    words; the import lines may not."""
    imports = [ln for ln in SCRIPT.read_text().splitlines()
               if ln.startswith(("import ", "from "))]
    assert "from jarvis import contacts as book_mod                   # noqa: E402" in imports
    for line in imports:
        assert "assistant_config" not in line and "AssistantConfig" not in line
        assert "cmdsock" not in line and "socket" not in line
        assert "identity" not in line and "memory" not in line


def test_cli_list_on_an_empty_book(cli, book_file, capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "0 people in" in out and "contacts.json" in out


def test_cli_add_then_list_then_json(cli, book_file, capsys):
    assert cli.main(["add", "Heather Smith", "heather@example.com", "--honorific",
                     "Dr", "--alias", "my advisor", "--note", "PhD advisor"]) == 0
    assert capsys.readouterr().out.strip() == "added Heather Smith <heather@example.com>"
    assert stat.S_IMODE(book_file.stat().st_mode) == 0o600
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Heather Smith  <heather@example.com>  Dr  aliases: my advisor  " \
           "note: PhD advisor" in out
    assert out.strip().endswith("1 person in " + book_mod.display_path(book_file))
    assert cli.main(["list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["contacts"][0]["email"] == "heather@example.com"
    assert data["skipped"] == []


def test_cli_add_refuses_a_bad_address_and_writes_nothing(cli, book_file, capsys):
    assert cli.main(["add", "Heather Smith", "heather at example dot com"]) == 2
    assert capsys.readouterr().out.startswith("REFUSED: bad address")
    assert not book_file.exists()
    assert cli.main(["add", "heather@example.com", "Heather Smith"]) == 2
    assert "wrong box" in capsys.readouterr().out
    assert not book_file.exists()


def test_cli_add_refuses_a_duplicate(cli, two_heathers, capsys):
    assert cli.main(["add", "heather smith", "new@example.com"]) == 2
    assert capsys.readouterr().out == "REFUSED: already in the book as Heather Smith\n"
    assert cli.main(["add", "Someone New", "HJONES@example.com"]) == 2
    assert "already in the book as Heather Jones" in capsys.readouterr().out


def test_cli_show_is_the_send_lanes_own_verdict(cli, two_heathers, capsys):
    assert cli.main(["show", "Heather Smith"]) == 0
    assert capsys.readouterr().out.strip() == \
        "FOUND: Heather Smith <heather@example.com> (matched on full name)"
    assert cli.main(["show", "mom"]) == 0
    assert "(matched on alias)" in capsys.readouterr().out
    assert cli.main(["show", "Heather"]) == 3
    out = capsys.readouterr().out
    assert out.startswith("AMBIGUOUS: Heather Smith, Heather Jones - Jarvis would ask")
    assert "Which Heather, sir" in out
    assert cli.main(["show", "Heathr"]) == 2
    assert capsys.readouterr().out.strip() == "UNKNOWN: nothing in the book for 'Heathr'"


def test_cli_remove_needs_a_terminal_or_yes(cli, two_heathers, capsys):
    assert cli.main(["remove", "Heather Jones"]) == 2
    out = capsys.readouterr().out
    assert "REFUSED: removing needs a terminal to confirm, or --yes" in out
    assert len(json.loads(two_heathers.read_text())["contacts"]) == 3
    assert cli.main(["remove", "Jones", "--yes"]) == 0
    assert capsys.readouterr().out.strip().endswith(
        "removed Heather Jones <hjones@example.com>")
    assert [r["name"] for r in json.loads(two_heathers.read_text())["contacts"]] == \
        ["Heather Smith", "Mum"]


def test_cli_remove_refuses_an_ambiguous_or_unknown_name(cli, two_heathers, capsys):
    assert cli.main(["remove", "Heather", "--yes"]) == 2
    assert capsys.readouterr().out.strip() == \
        "REFUSED: which one - Heather Smith or Heather Jones? Give the full name"
    assert cli.main(["remove", "Heathr", "--yes"]) == 2
    assert "REFUSED: nothing in the book" in capsys.readouterr().out
    assert len(json.loads(two_heathers.read_text())["contacts"]) == 3


def test_cli_remove_at_a_terminal_asks_and_takes_only_a_yes(cli, two_heathers,
                                                            capsys, monkeypatch):
    monkeypatch.setattr(cli, "_isatty", lambda s: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(["remove", "Mum"]) == 2
    assert "nothing removed" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert cli.main(["remove", "Mum"]) == 0
    assert "removed Mum <linda@example.com>" in capsys.readouterr().out


def test_cli_list_flags_a_bad_row_last(cli, book_file, capsys):
    write(book_file, [{"name": "Dana Ruiz", "email": "dana@example"}, TWO_HEATHERS[0]])
    assert cli.main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Heather Smith")
    assert lines[1].startswith("BAD (not used): row 1 Dana Ruiz: bad address")
    assert lines[-1].startswith("1 person in")


def test_cli_cannot_write_is_exit_1(cli, two_heathers, capsys, monkeypatch):
    monkeypatch.setattr(book_mod, "_write_private",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError("ro")))
    assert cli.main(["add", "Dana Ruiz", "dana@example.com"]) == 1
    assert "could not be written" in capsys.readouterr().out


# ==================================================================
# 7. The review's four: a broken file is never written over, a one-word
#    name that is also someone's first name is a question, an ordinal
#    only counts when the list was read, and the address check is what
#    the docs say it is
# ==================================================================
import subprocess  # noqa: E402
import sys  # noqa: E402

BROKEN = "contacts.json is not valid JSON"
FIX_IT = "fix it by hand first"


def _corrupt(path: Path) -> str:
    """A 2-row book with the trailing comma a hand edit leaves behind."""
    text = ('{\n  "format": 1,\n  "contacts": [\n'
            '    {"name": "Heather Smith", "email": "heather@example.com"},\n'
            '    {"name": "Heather Jones", "email": "hjones@example.com"},\n'
            '  ]\n}\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ---- (1) silent book loss
def test_a_broken_file_refuses_an_add_and_is_not_touched_in_process(two_heathers):
    """The last-good rows stay for RESOLVING; nothing writes them back."""
    book = book_mod.current()
    assert book.resolve("Heather Smith").found
    text = _corrupt(two_heathers)
    before = two_heathers.stat()
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is None
    assert why.startswith(BROKEN) and str(two_heathers) in why and why.endswith(FIX_IT)
    assert two_heathers.read_text(encoding="utf-8") == text, "byte-identical"
    assert two_heathers.stat().st_mtime_ns == before.st_mtime_ns
    # resolving still works off the last good book
    assert book.resolve("Heather Smith").found
    # the last good rows are NOT what remove works on either
    gone, why = book.remove("Heather Smith", "heather@example.com")
    assert gone is None and why.startswith(BROKEN)
    assert two_heathers.read_text(encoding="utf-8") == text


def test_a_broken_file_in_a_fresh_process_refuses_the_cli_add(book_file):
    """The reproduced loss: a fresh process has NO last-good rows, so the
    old code wrote [the new row] over his two. Run the real script."""
    text = _corrupt(book_file)
    env = dict(os.environ, **{book_mod.ENV_VAR: str(book_file)})
    proc = subprocess.run([sys.executable, str(SCRIPT), "add", "Dana Ruiz",
                           "dana@example.com"], env=env, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert proc.stdout.startswith("REFUSED: " + BROKEN), proc.stdout
    assert str(book_file) in proc.stdout and proc.stdout.rstrip().endswith(FIX_IT)
    assert book_file.read_text(encoding="utf-8") == text, "byte-identical"
    proc = subprocess.run([sys.executable, str(SCRIPT), "remove", "Heather Smith",
                           "--yes"], env=env, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 2
    assert proc.stdout.startswith("REFUSED: " + BROKEN), proc.stdout
    assert proc.stdout.rstrip().endswith(FIX_IT), "remove says the file is broken"
    assert book_file.read_text(encoding="utf-8") == text


def test_the_cli_refusal_line_is_the_one_the_docs_promise(cli, two_heathers, capsys):
    _corrupt(two_heathers)
    assert cli.main(["add", "Dana Ruiz", "dana@example.com"]) == 2
    out = capsys.readouterr().out.strip()
    assert out == (f"REFUSED: contacts.json is not valid JSON ({two_heathers}) "
                   "— fix it by hand first")


@pytest.mark.parametrize("text", ['{"format": 2, "contacts": []}',
                                  '{"contacts": "nope", "format": 1}', '[]'])
def test_a_wrong_shape_refuses_a_write_too(two_heathers, text):
    book = book_mod.current()
    two_heathers.write_text(text, encoding="utf-8")
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is None and "contacts.json" in why and why.endswith(FIX_IT)
    assert two_heathers.read_text(encoding="utf-8") == text


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads anything")
def test_an_unreadable_file_refuses_a_write(two_heathers):
    import time
    book = book_mod.current()
    assert book.resolve("Mum").found
    # Kernel file times are coarse (a tick of a few ms): a chmod in the same
    # tick as the write leaves ctime where it was. One tick between them is
    # how it happens in life; the stamp carries ctime so a chmod is seen.
    time.sleep(0.05)
    two_heathers.chmod(0)
    try:
        contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
        assert contact is None
        assert "contacts.json could not be read" in why and why.endswith(FIX_IT)
        assert book.resolve("Mum").found, "the last good book still resolves"
        # a fresh process has no last good book: still a refusal, still nothing
        fresh = book_mod.Book(two_heathers)
        assert fresh.resolve("Mum").unknown
        assert fresh.add({"name": "Dana Ruiz", "email": "dana@example.com"})[0] is None
    finally:
        two_heathers.chmod(0o600)
    # and once it is readable again the write goes through
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is not None and why == ""


def test_a_fixed_file_writes_again(two_heathers):
    book = book_mod.current()
    _corrupt(two_heathers)
    assert book.add({"name": "Dana Ruiz", "email": "dana@example.com"})[0] is None
    write(two_heathers, TWO_HEATHERS)
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is not None and why == ""
    assert [r["name"] for r in json.loads(two_heathers.read_text())["contacts"]] == \
        ["Heather Smith", "Heather Jones", "Mum", "Dana Ruiz"]


# ---- (2) a one-word name that is also someone's first name or surname
@pytest.mark.parametrize("existing, new, why_has", [
    ({"name": "Heather Jones", "email": "hj@example.com"},
     {"name": "Heather", "email": "h@example.com"}, "first name"),
    ({"name": "Heather", "email": "h@example.com"},
     {"name": "Heather Jones", "email": "hj@example.com"}, "first name"),
    ({"name": "Sam Smith", "email": "ss@example.com"},
     {"name": "Smith", "email": "s@example.com"}, "surname"),
    ({"name": "Smith", "email": "s@example.com"},
     {"name": "Sam Smith", "email": "ss@example.com"}, "surname"),
    ({"name": "Heather Smith", "email": "hs@example.com", "honorific": "Dr"},
     {"name": "Dr Smith", "email": "d@example.com"}, "Dr"),
    ({"name": "Linda Mum", "email": "l@example.com"},
     {"name": "Mum", "email": "m@example.com"}, "surname"),
])
def test_a_one_word_name_may_not_be_another_rows_first_name_or_surname(
        existing, new, why_has):
    old, _ = book_mod.validate_row(existing)
    row, _ = book_mod.validate_row(new)
    why = book_mod.check_unique([old], row)
    assert why, (existing, new)
    assert why_has in why, why
    assert old.name in why


def test_two_multi_word_names_that_share_a_first_name_or_surname_are_still_fine():
    a, _ = book_mod.validate_row({"name": "Heather Smith", "email": "a@example.com"})
    b, _ = book_mod.validate_row({"name": "Heather Jones", "email": "b@example.com"})
    c, _ = book_mod.validate_row({"name": "John Smith", "email": "c@example.com"})
    assert book_mod.check_unique([a], b) == ""
    assert book_mod.check_unique([a], c) == ""


def test_a_one_word_collision_is_refused_on_add_and_writes_nothing(book_file):
    write(book_file, [{"name": "Heather Jones", "email": "hj@example.com"}])
    before = book_file.read_text()
    contact, why = book_mod.current().add({"name": "Heather", "email": "h@example.com"})
    assert contact is None and "first name" in why
    assert book_file.read_text() == before


def test_a_one_word_collision_on_load_keeps_both_flags_both_and_asks(book_file, caplog):
    write(book_file, [{"name": "Heather", "email": "h@example.com"},
                      {"name": "Heather Jones", "email": "hj@example.com"},
                      {"name": "Mum", "email": "linda@example.com"}])
    with caplog.at_level(logging.WARNING, logger="contacts"):
        book = book_mod.current()
    assert [c.name for c in book.contacts] == ["Heather", "Heather Jones", "Mum"], \
        "both rows are kept"
    assert book.skipped == []
    assert sorted(f.name for f in book.flagged) == ["Heather", "Heather Jones"]
    assert all("question" in f.why for f in book.flagged)
    assert "example.com" not in caplog.text
    # the collision resolves as a QUESTION, never a pick
    res = book.resolve("Heather")
    assert res.ambiguous and res.candidates == ["Heather", "Heather Jones"], res
    # the full two-word name is still exact
    assert book.resolve("Heather Jones").addr == "hj@example.com"
    assert book.resolve("Jones").addr == "hj@example.com"
    # a one-word row that is unique reads back fine
    assert book.resolve("Mum").addr == "linda@example.com"
    # and the which-answer cannot be "Heather" -- that is the question
    cands = ["Heather", "Heather Jones"]
    assert book.choose("Heather", cands) is None
    assert book.choose("Heather Jones", cands) == "Heather Jones"
    assert book.choose("Jones", cands) == "Heather Jones"
    assert "flagged" in book.public() and len(book.public()["flagged"]) == 2


def test_a_shared_surname_with_a_one_word_row_is_a_question(book_file):
    write(book_file, [{"name": "Sam Smith", "email": "ss@example.com"},
                      {"name": "Smith", "email": "s@example.com"}])
    book = book_mod.current()
    assert book.resolve("Smith").ambiguous
    assert book.resolve("Smith").candidates == ["Sam Smith", "Smith"]
    assert book.resolve("Sam Smith").addr == "ss@example.com"
    assert book.resolve("Sam").addr == "ss@example.com"


def test_the_commander_asks_which_heather_for_a_one_word_collision(cmd_book):
    write(book_mod.book_path(), [{"name": "Heather", "email": "h@example.com"},
                                 {"name": "Heather Jones", "email": "hj@example.com"}])
    res = cmd_book.handle("email the biosensors handout to Heather", source="voice")
    assert res.reply == "Which Heather, sir — Heather or Heather Jones?"
    assert cmd_book._pending_send is None
    res = cmd_book.handle("Heather", source="voice")
    assert res.reply.startswith("The full name, sir"), res
    res = cmd_book.handle("Jones", source="voice")
    assert "to Heather Jones, from your school account" in res.reply
    assert res.display_only == "to hj@example.com"
    assert not FakeSMTP.made


def test_cli_list_shows_the_collision_as_kept_and_flagged(cli, book_file, capsys):
    write(book_file, [{"name": "Heather", "email": "h@example.com"},
                      {"name": "Heather Jones", "email": "hj@example.com"}])
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "2 people in" in out
    assert "AMBIGUOUS (kept): row 1 Heather:" in out
    assert "AMBIGUOUS (kept): row 2 Heather Jones:" in out
    assert cli.main(["show", "Heather"]) == 3


# ---- (3) an ordinal only counts when the list was actually read
FIVE = [{"name": f"Heather {s}", "email": f"h{s.lower()}@example.com"}
        for s in ("Adams", "Brown", "Clark", "Davis", "Evans")]


@pytest.mark.parametrize("said", ["the first one", "the second one", "2", "5",
                                  "the last one", "the other one"])
def test_an_ordinal_is_a_miss_when_the_list_was_not_read(said):
    from jarvis import commander as cm
    names = [r["name"] for r in FIVE]
    assert len(names) > book_mod.WHICH_MAX
    assert cm._person_from_answer(said, names) is None, said
    # and it still works for a list that WAS read
    four = names[:book_mod.WHICH_MAX]
    assert cm._person_from_answer("the first one", four) == "Heather Adams"
    assert cm._person_from_answer("the last one", four) == "Heather Davis"


def test_five_heathers_the_first_one_is_re_asked_for_the_full_name(cmd_book):
    write(book_mod.book_path(), FIVE)
    res = cmd_book.handle("email the biosensors handout to Heather", source="voice")
    assert res.reply == "I've five people called Heather, sir — the full name, please."
    res = cmd_book.handle("the first one", source="voice")
    assert res.reply.startswith("The full name"), res
    assert cmd_book._pending_send is None and cmd_book.question_open()
    res = cmd_book.handle("Heather Clark", source="voice")
    assert "to Heather Clark, from your school account" in res.reply
    assert res.display_only == "to hclark@example.com"
    assert not FakeSMTP.made


# ---- (4) the address check is what the docs say it is
@pytest.mark.parametrize("bad", [
    "x@-.-", "a@b.c", "h@1.2", "h@example.com-", "heather..x@example.com",
    ".h@example.com", "h.@example.com", "h@-example.com", "h@example-.com",
    "h@example.c0m", "h@example.", "h@.example.com", "h@" + "a" * 64 + ".com",
    "h@example.com.", "h" + "@" + "x" * 250 + ".com",
])
def test_the_reviewers_accepted_bad_addresses_are_refused(bad):
    contact, why = book_mod.validate_row({"name": "Heather Smith", "email": bad})
    assert contact is None, bad
    assert why.startswith("bad address") or "longer than" in why, why


@pytest.mark.parametrize("good", [
    "heather@example.com", "heather.smith@example.co.uk", "h+tag@example.com",
    "h_s%x@sub-domain.example.org", "h@" + "a" * 63 + ".com", "H@EXAMPLE.COM",
    "heather@gmail.con",
])
def test_a_well_formed_address_is_accepted_even_when_it_is_wrong(good):
    contact, why = book_mod.validate_row({"name": "Heather Smith", "email": good})
    assert contact is not None, (good, why)


def test_the_docs_and_the_cli_help_admit_gmail_con_cannot_be_caught(cli, capsys):
    doc = (Path(__file__).resolve().parents[1] / "docs" / "assistant-setup.md"
           ).read_text(encoding="utf-8")
    assert "gmail.con" in doc and "read-back" in doc
    with pytest.raises(SystemExit):
        cli.main(["add", "--help"])
    assert "gmail.con" in capsys.readouterr().out


# ==================================================================
# 7. The re-review (round 2): a one-word row he can actually choose,
#    "Dr Heather" is the same question, a symlinked book stays one, the
#    CLI says "broken" when it is, and four small leftovers
# ==================================================================
ONE_WORD = [{"name": "Heather", "email": "h@example.com"},
            {"name": "Heather Jones", "email": "hj@example.com"}]


# ---- (B1) the ordinal picks the row of the list that was READ, by identity
@pytest.mark.parametrize("said, who, addr", [
    ("the first one", "Heather", "h@example.com"),
    ("1", "Heather", "h@example.com"),
    ("number one", "Heather", "h@example.com"),
    ("the second one", "Heather Jones", "hj@example.com"),
    ("2", "Heather Jones", "hj@example.com"),
    ("Jones", "Heather Jones", "hj@example.com"),
    ("Heather Jones", "Heather Jones", "hj@example.com"),
])
def test_an_ordinal_picks_the_one_word_row_of_the_list_that_was_read(
        cmd_book, said, who, addr):
    """The reviewed loop: "the first one" re-resolved the spoken name
    "Heather", the collision rule asked again, and a FRESH slot was made
    every time -- the same question four times running, never a pick,
    never the drop. The answer names a row of the list he was read; it is
    resolved by that row's identity and never back through the name."""
    write(book_mod.book_path(), ONE_WORD)
    res = cmd_book.handle("email the biosensors handout to Heather", source="voice")
    assert res.reply == "Which Heather, sir — Heather or Heather Jones?"
    res = cmd_book.handle(said, source="voice")
    assert res is not None and res.reply.endswith("Send it, sir?"), (said, res)
    assert f"to {who}, from your school account" in res.reply
    assert "example" not in res.reply, "the address is never spoken"
    assert res.display_only == f"to {addr}", "and it is shown"
    assert cmd_book._pending_send is not None
    assert cmd_book._pending_sendask is None
    assert not FakeSMTP.made


@pytest.mark.parametrize("said, who, addr", [
    ("the second one", "Mum", "linda@example.com"),
    ("2", "Mum", "linda@example.com"),
    ("the first one", "Sam Mum", "sam@example.com"),
    ("Sam Mum", "Sam Mum", "sam@example.com"),
])
def test_sam_mum_and_mum_the_ordinal_picks_and_the_yes_sends_there(cmd_book, said,
                                                                   who, addr):
    write(book_mod.book_path(), [{"name": "Sam Mum", "email": "sam@example.com"},
                                 {"name": "Mum", "email": "linda@example.com"}])
    res = cmd_book.handle("email the biosensors handout to Mum", source="voice")
    assert res.reply == "Which Mum, sir — Sam Mum or Mum?"
    res = cmd_book.handle(said, source="voice")
    assert f"to {who}, from your school account" in res.reply, res
    assert res.display_only == f"to {addr}"
    assert cmd_book._pending_send is not None and not FakeSMTP.made
    res = cmd_book.handle("yes", source="voice")
    assert res.ack and FakeSMTP.made[-1].sent[0]["To"] == addr
    assert cmd_book.spoken == [f"Sent to {who}, sir."]


def test_a_miss_spends_the_same_question_and_an_ordinal_then_still_picks(cmd_book):
    write(book_mod.book_path(), ONE_WORD)
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    ask = cmd_book._pending_sendask
    assert ask is not None and ask.kind == "person" and ask.reasked is False
    res = cmd_book.handle("Heather", source="voice")     # the ambiguity itself
    assert res.reply == "The full name, sir — Heather or Heather Jones?"
    assert cmd_book._pending_sendask is ask, "the SAME slot, not a fresh one"
    assert ask.reasked is True
    res = cmd_book.handle("the first one", source="voice")
    assert "to Heather, from your school account" in res.reply, res
    assert res.display_only == "to h@example.com"
    assert cmd_book._pending_sendask is None and not FakeSMTP.made


def test_two_misses_are_the_drop_never_the_same_question_again(cmd_book):
    write(book_mod.book_path(), ONE_WORD)
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    ask = cmd_book._pending_sendask
    res = cmd_book.handle("Heather", source="voice")
    assert res.reply.startswith("The full name, sir") and cmd_book._pending_sendask is ask
    res = cmd_book.handle("Heather", source="voice")
    assert res.reply == outbox.ASK_DROPPED_LINE and res.status == "Dropped"
    assert cmd_book._pending_sendask is None and cmd_book._pending_send is None
    # with no question on the floor an ordinal answers nothing
    res = cmd_book.handle("the first one", source="voice")
    assert res is None or not str(res.reply or "").startswith("Which Heather")
    assert cmd_book._pending_send is None and cmd_book._pending_sendask is None
    assert not FakeSMTP.made


def test_a_row_that_went_between_the_question_and_the_answer_is_a_miss(cmd_book):
    """He was read a list; by the time he answers the row is gone (a hand
    edit). Re-resolving the NAME would quietly find the other Heather; by
    identity it is nobody, so it is a miss and the question is spent."""
    write(book_mod.book_path(), ONE_WORD)
    cmd_book.handle("email the biosensors handout to Heather", source="voice")
    ask = cmd_book._pending_sendask
    write(book_mod.book_path(), ONE_WORD[1:])
    res = cmd_book.handle("the first one", source="voice")
    assert res.reply.startswith("The full name, sir"), res
    assert cmd_book._pending_sendask is ask and ask.reasked is True
    assert cmd_book._pending_send is None and not FakeSMTP.made


def test_pick_is_by_identity_and_never_a_question(book_file):
    write(book_file, ONE_WORD)
    book = book_mod.current()
    assert book.resolve("Heather").ambiguous, "the name is the question"
    res = book.pick("Heather")
    assert res is not None and res.found and res.addr == "h@example.com"
    assert res.name == "Heather" and res.from_book and res.matched_on == "full name"
    assert book.pick("Heather Jones").addr == "hj@example.com"
    assert book.pick("Jones") is None, "a full name only, never a part"
    assert book.pick("Nobody") is None


# ---- (A3) honorific + a flagged one-word name is the same question
def test_dr_heather_against_a_flagged_one_word_row_is_a_question(book_file):
    write(book_file, [{"name": "Heather", "email": "h@example.com", "honorific": "Dr"},
                      {"name": "Heather Jones", "email": "hj@example.com"}])
    book = book_mod.current()
    for said in ("Dr Heather", "Dr. Heather", "dr heather"):
        res = book.resolve(said)
        assert res.ambiguous and res.candidates == ["Heather", "Heather Jones"], (said, res)
    # the two-word name and its honorific form stay exact
    assert book.resolve("Heather Jones").addr == "hj@example.com"
    assert book.resolve("Jones").addr == "hj@example.com"
    # the honorific on both rows changes nothing
    write(book_file, [{"name": "Heather", "email": "h@example.com", "honorific": "Dr"},
                      {"name": "Heather Jones", "email": "hj@example.com",
                       "honorific": "Dr"}])
    assert book.resolve("Dr Heather").candidates == ["Heather", "Heather Jones"]
    assert book.resolve("Dr Heather Jones").addr == "hj@example.com"
    # and as the ANSWER to the question it is the ambiguity again, not a pick
    cands = ["Heather", "Heather Jones"]
    assert book.choose("Dr Heather", cands) is None
    assert book.choose("Dr Heather Jones", cands) == "Heather Jones"
    assert book.choose("Jones", cands) == "Heather Jones"


def test_control_dr_smith_beside_dr_heather_smith_still_asks(book_file):
    write(book_file, [{"name": "Smith", "email": "s@example.com", "honorific": "Dr"},
                      {"name": "Heather Smith", "email": "hs@example.com",
                       "honorific": "Dr"}])
    book = book_mod.current()
    assert book.resolve("Dr Smith").candidates == ["Smith", "Heather Smith"]
    assert book.resolve("Smith").candidates == ["Smith", "Heather Smith"]
    assert book.resolve("Dr Heather Smith").addr == "hs@example.com"
    assert book.resolve("Heather").addr == "hs@example.com"


def test_an_exact_two_word_name_with_its_honorific_still_wins(two_heathers):
    book = book_mod.current()
    assert book.resolve("Dr Heather Smith").addr == "heather@example.com"
    assert book.resolve("Dr Smith").addr == "heather@example.com"


def test_the_commander_asks_for_dr_heather_and_the_ordinal_picks(cmd_book):
    write(book_mod.book_path(), [{"name": "Heather", "email": "h@example.com",
                                  "honorific": "Dr"},
                                 {"name": "Heather Jones", "email": "hj@example.com"}])
    res = cmd_book.handle("email the biosensors handout to Dr Heather", source="voice")
    assert res.reply.startswith("Which") and res.reply.endswith("Heather or Heather Jones?"), res
    assert res.status == "Which person?" and cmd_book._pending_send is None
    res = cmd_book.handle("the first one", source="voice")
    assert "to Dr Heather, from your school account" in res.reply, res
    assert res.display_only == "to h@example.com"
    assert not FakeSMTP.made


# ---- (A2) a symlinked book stays a symlink; the write lands beside the target
def _no_temp_litter(*dirs):
    return all(not list(Path(d).glob(".contacts-*")) for d in dirs)


def test_a_symlinked_book_stays_a_symlink_and_the_target_gets_the_row(tmp_path, book_file):
    target = tmp_path / "dotfiles" / "book.json"
    write(target, [{"name": "Mum", "email": "linda@example.com"}])
    book_file.parent.mkdir(parents=True, exist_ok=True)
    book_file.symlink_to(target)
    book = book_mod.current()
    assert book.resolve("Mum").found
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is not None, why
    assert book_file.is_symlink(), "the link is replaced by a regular file"
    assert Path(os.readlink(book_file)) == target
    names = [r["name"] for r in json.loads(target.read_text())["contacts"]]
    assert names == ["Mum", "Dana Ruiz"], "the row went to the REAL file"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert _no_temp_litter(book_file.parent, target.parent)
    # a hand edit of the target is live on the next resolve, no restart
    data = json.loads(target.read_text())
    data["contacts"].append({"name": "Zed Zorro", "email": "zed@example.com"})
    target.write_text(json.dumps(data), encoding="utf-8")
    assert book.resolve("Zed Zorro").addr == "zed@example.com"
    assert book.resolve("Dana Ruiz").addr == "dana@example.com"
    # remove writes beside the target too
    gone, why = book.remove("Mum", "linda@example.com")
    assert gone is not None and book_file.is_symlink()
    names = [r["name"] for r in json.loads(target.read_text())["contacts"]]
    assert names == ["Dana Ruiz", "Zed Zorro"]


def test_a_dangling_symlink_gets_its_target_created(tmp_path, book_file):
    target = tmp_path / "dotfiles" / "book.json"        # dotfiles/ does not exist
    book_file.parent.mkdir(parents=True, exist_ok=True)
    book_file.symlink_to(target)
    book = book_mod.current()
    assert book.broken == "" and book.contacts == [], "a missing target is an empty book"
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is not None, why
    assert book_file.is_symlink() and target.is_file()
    assert [r["name"] for r in json.loads(target.read_text())["contacts"]] == ["Dana Ruiz"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert book.resolve("Dana Ruiz").found


# ---- (A1) the CLI says the book is broken, not "nothing in the book"
def test_cli_remove_and_show_on_a_broken_file_say_the_promised_line(cli, two_heathers, capsys):
    text = _corrupt(two_heathers)
    line = (f"REFUSED: contacts.json is not valid JSON ({two_heathers}) "
            "— fix it by hand first")
    assert cli.main(["remove", "Heather Smith", "--yes"]) == 2
    assert capsys.readouterr().out.strip() == line
    assert cli.main(["show", "Heather Smith"]) == 2
    out = capsys.readouterr().out.strip()
    assert out == line, out
    assert "UNKNOWN" not in out and "nothing in the book" not in out
    assert two_heathers.read_text(encoding="utf-8") == text


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads anything")
def test_cli_remove_and_show_on_an_unreadable_file_say_so(cli, two_heathers, capsys):
    two_heathers.chmod(0)
    try:
        line = (f"REFUSED: contacts.json could not be read ({two_heathers}) "
                "— fix it by hand first")
        assert cli.main(["remove", "Mum", "--yes"]) == 2
        assert capsys.readouterr().out.strip() == line
        assert cli.main(["show", "Mum"]) == 2
        assert capsys.readouterr().out.strip() == line
    finally:
        two_heathers.chmod(0o600)


# ---- (a) empty is empty, a BOM is a BOM, anything else is broken
@pytest.mark.parametrize("text", ["", "\n", "   \n\t \n"])
def test_an_empty_or_whitespace_file_is_an_empty_writable_book(book_file, text):
    """`touch contacts.json` then `add` must work: a 0-byte file is what a
    missing file is, an empty book, not a broken one."""
    book_file.parent.mkdir(parents=True, exist_ok=True)
    book_file.write_text(text, encoding="utf-8")
    book = book_mod.current()
    assert book.broken == "" and book.contacts == [] and book.skipped == []
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is not None, why
    data = json.loads(book_file.read_text(encoding="utf-8"))
    assert data == {"format": 1, "contacts": [{"name": "Dana Ruiz",
                                              "email": "dana@example.com"}]}


def test_a_utf8_bom_is_stripped_on_read(book_file):
    body = json.dumps({"format": 1, "contacts": [
        {"name": "Mum", "email": "linda@example.com"}], "comment": "kept"})
    book_file.parent.mkdir(parents=True, exist_ok=True)
    book_file.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    book = book_mod.current()
    assert book.broken == "", book.broken
    assert book.resolve("Mum").addr == "linda@example.com"
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is not None, why
    raw = book_file.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "the rewrite is plain UTF-8"
    data = json.loads(raw.decode("utf-8"))
    assert [r["name"] for r in data["contacts"]] == ["Mum", "Dana Ruiz"]
    assert data["comment"] == "kept"


@pytest.mark.parametrize("text", ["{", "[]", "null", "42", '"book"', "{,}",
                                  '{"format": 1, "contacts": [}',
                                  "﻿{"])
def test_anything_else_malformed_stays_broken(book_file, text):
    book_file.parent.mkdir(parents=True, exist_ok=True)
    book_file.write_text(text, encoding="utf-8")
    book = book_mod.current()
    assert book.broken.startswith("contacts.json") and book.broken.endswith(FIX_IT), text
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is None and why == book.broken
    assert book_file.read_text(encoding="utf-8") == text


# ---- (c) an alias that is someone's name loses the ALIAS, not a person
@pytest.mark.parametrize("order", ["alias row first", "name row first"])
def test_an_alias_that_is_someones_name_drops_the_alias_and_keeps_both_people(
        book_file, order):
    mum = {"name": "Mum", "email": "linda@example.com", "aliases": ["mom", "Heather"]}
    heather = {"name": "Heather", "email": "h@example.com"}
    rows = [mum, heather] if order == "alias row first" else [heather, mum]
    write(book_file, rows)
    before = book_file.read_text(encoding="utf-8")
    book = book_mod.current()
    assert sorted(c.name for c in book.contacts) == ["Heather", "Mum"], "both kept"
    assert book.skipped == [] and book.flagged == []
    assert book.resolve("Heather").addr == "h@example.com", "the NAME wins"
    assert book.resolve("Mum").addr == "linda@example.com"
    assert book.resolve("mom").addr == "linda@example.com", "the other alias stays"
    assert len(book.trimmed) == 1
    t = book.trimmed[0]
    assert t.name == "Mum" and t.index == rows.index(mum)
    assert "Heather" in t.why and "alias" in t.why.lower(), t.why
    assert "example.com" not in t.why
    assert book.public()["trimmed"] == [{"index": t.index, "name": "Mum", "why": t.why}]
    assert book_file.read_text(encoding="utf-8") == before, "a load writes nothing"
    # a rewrite keeps HIS alias in the file: the file is his
    book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    kept = [r for r in json.loads(book_file.read_text())["contacts"] if r["name"] == "Mum"]
    assert kept[0]["aliases"] == ["mom", "Heather"]


def test_an_alias_that_is_a_first_name_or_another_alias_is_trimmed_too(book_file):
    write(book_file, [{"name": "Heather Jones", "email": "hj@example.com"},
                      {"name": "Mum", "email": "linda@example.com",
                       "aliases": ["Heather", "mom"]},
                      {"name": "Dad", "email": "dad@example.com", "aliases": ["mom", "pa"]}])
    book = book_mod.current()
    assert [c.name for c in book.contacts] == ["Heather Jones", "Mum", "Dad"]
    assert book.resolve("Heather").addr == "hj@example.com"
    assert book.resolve("mom").addr == "linda@example.com", "first in the file keeps it"
    assert book.resolve("pa").addr == "dad@example.com"
    assert [(t.name, t.index) for t in book.trimmed] == [("Mum", 1), ("Dad", 2)]


def test_cli_list_and_the_page_show_the_dropped_alias(cli, book_file, capsys):
    write(book_file, [{"name": "Heather", "email": "h@example.com"},
                      {"name": "Mum", "email": "linda@example.com", "aliases": ["Heather"]}])
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "2 people in" in out
    assert "row 2 Mum:" in out and "Heather" in out.split("row 2 Mum:")[1]
    assert "(kept)" in out and "alias" in out.lower()
    assert cli.main(["show", "Heather"]) == 0
    assert "FOUND: Heather <h@example.com>" in capsys.readouterr().out


def test_add_still_refuses_an_alias_that_is_someones_name(two_heathers):
    book = book_mod.current()
    before = two_heathers.read_text(encoding="utf-8")
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com",
                             "aliases": ["Heather"]})
    assert contact is None and "already how" in why
    contact, why = book.add({"name": "Advisor", "email": "dana@example.com"})
    assert contact is None and "already an alias" in why
    assert two_heathers.read_text(encoding="utf-8") == before


# ---- (f) the commander's INFO lines never carry an address he typed
def test_the_commander_log_masks_an_address_he_typed(cmd_book, caplog):
    with caplog.at_level(logging.INFO, logger="jarvis.commander"):
        cmd_book.handle("email the biosensors handout to Heather Smith", source="voice")
        cmd_book.handle("yes, to hjones@example.com", source="voice")
        cmd_book.handle("no", source="voice")
        cmd_book.handle("email the biosensors handout to dana@example.com",
                        source="voice")
    lines = [r.getMessage() for r in caplog.records if r.name == "jarvis.commander"]
    assert lines, "the commander logged nothing at INFO"
    assert "hjones@example.com" not in caplog.text
    assert "dana@example.com" not in caplog.text
    handled = [ln for ln in lines if ln.startswith("handle ")]
    assert any("h…@example.com" in ln for ln in handled), handled
    assert any("d…@example.com" in ln for ln in handled), handled
    readback = [ln for ln in lines if ln.startswith("send read-back:")
                and "corrects the draft" in ln]
    assert readback and all("h…@example.com" in ln for ln in readback), readback
    assert not FakeSMTP.made


def test_mask_addresses_masks_every_address_in_a_sentence():
    assert outbox.mask_addresses("yes, to hjones@example.com") == "yes, to h…@example.com"
    assert outbox.mask_addresses("cc a@x.org and b@y.co.uk.") == "cc a…@x.org and b…@y.co.uk."
    assert outbox.mask_addresses("no address here") == "no address here"
    assert outbox.mask_addresses("") == ""


# ---- (g) the address he SAYS is masked too. "dana at example dot com" is
#      how an address is said, and how Whisper writes it far more often
#      than it writes the symbols; the typed mask keyed on "@" and left
#      every spoken address verbatim in the commander's INFO lines.
#      Every name and domain here is invented.
@pytest.mark.parametrize("said, want", [
    ("dana at example dot com", "d… at example dot com"),
    ("email the handout to dana at example dot com",
     "email the handout to d… at example dot com"),
    ("Dana at Example dot Com", "D… at Example dot Com"),
    ("DANA AT EXAMPLE DOT COM", "D… AT EXAMPLE DOT COM"),
    ("dana at gmail dot com", "d… at gmail dot com"),
    ("it's dana at gmail dot com.", "it's d… at gmail dot com."),
    ("dana at mail dot tamu dot edu", "d… at mail dot tamu dot edu"),
    ("dana at example dot co dot uk", "d… at example dot co dot uk"),
    ("dana at example period com", "d… at example period com"),
    ("Dana at gmail. com", "D… at gmail. com"),
    ("dana at gmail.com", "d… at gmail.com"),
    ("dana dot ruiz at example dot com", "d… at example dot com"),
    ("d dot ruiz at example dot com", "d… at example dot com"),
    ("dana underscore ruiz at example dot com", "d… at example dot com"),
    ("dana dash ruiz at example dot com", "d… at example dot com"),
    ("dana_ruiz at example dot com", "d… at example dot com"),
    ("dana99 at example dot com", "d… at example dot com"),
    ("her address is dana at example dot com, please",
     "her address is d… at example dot com, please"),
    ("send it to dana at example dot com instead",
     "send it to d… at example dot com instead"),
    ("dana at example dot com and sam at example dot org",
     "d… at example dot com and s… at example dot org"),
    ("dana@example.com or dana at example dot com",
     "d…@example.com or d… at example dot com"),
    ("send read-back: 'dana at example dot com' corrects the draft",
     "send read-back: 'd… at example dot com' corrects the draft"),
    ("dana  at  example  dot  com", "d…  at  example  dot  com"),
    ("to Dana, at dana at example dot com", "to Dana, at d… at example dot com"),
    ("dana at example dot com.", "d… at example dot com."),
    ("Dana at Gmail dot com", "D… at Gmail dot com"),
    # every joiner confirm-shapes' address_span reads (b61fb12) is a
    # shape the file can be SENT to, so it is a shape this must mask
    ("heather hyphen smith at example dot com", "h… at example dot com"),
    ("heather under score smith at example dot com", "h… at example dot com"),
    ("heather full stop smith at example dot com", "h… at example dot com"),
    ("h underscore peyrovi at tamu dot edu", "h… at tamu dot edu"),
    ("heather at spark dash lab dot com", "h… at spark dash lab dot com"),
    ("heather at spark hyphen lab full stop com", "h… at spark hyphen lab full stop com"),
    ("yes, send it to heather underscore smith at example dot com",
     "yes, send it to h… at example dot com"),
])
def test_mask_addresses_masks_the_spoken_shape(said, want):
    assert outbox.mask_addresses(said) == want


@pytest.mark.parametrize("prose", [
    "meet at noon",
    "at the dot",
    "look at the dot on the map",
    "at 4 dot 30",
    "meet me at 4 dot 30",
    "the meeting is at 4 dot 30 pm",
    "aim at the red dot",
    "look at the dot com bubble",
    "I'm at home. See you at six.",
    "we stopped at noon. Then we left",
    "polka dot at the dance",
    "connect the dots at the end",
    "version 3 dot 12 at the latest",
    "she stared at the dot for a minute",
    "look at that dot there",
    "at dot",
    "dot at",
    "email the biosensors handout to Heather Smith",
    "dana at example",
    "look at the dash board dot com",
    "stand at the hyphen. Then read on",
    "what time is it",
    "",
])
def test_mask_addresses_leaves_ordinary_prose_alone(prose):
    assert outbox.mask_addresses(prose) == prose


@pytest.mark.parametrize("said, want", [
    # a top level that is also an English word is still an address when
    # it is SAID with "dot" ...
    ("dana at example dot in", "d… at example dot in"),
    ("dana at example dot me", "d… at example dot me"),
    ("dana at me dot com", "d… at me dot com"),
    ("look at me. In the morning", "look at me. In the morning"),
    # ... but after a full stop it is the next sentence, so a punctuated
    # domain has to end on a top level in use -- and "example.xyz" is
    # left alone (a typed address has an "@" and is masked by that)
    ("I'm at home. In the morning", "I'm at home. In the morning"),
    ("I'm at home. Info for you", "I'm at home. Info for you"),
    ("dana at example.xyz", "dana at example.xyz"),
    # a website said in a sentence has the same skeleton and is masked
    # too: a lost letter in a log line is cheaper than a leaked address
    ("the site is at example dot com", "the site i… at example dot com"),
    ("look at handout dot pdf", "l… at handout dot pdf"),
])
def test_mask_addresses_deliberate_calls(said, want):
    """Where the rule is a trade-off, this is the side it takes."""
    assert outbox.mask_addresses(said) == want


def _leaks(records, *raw):
    """Every captured record whose MESSAGE carries one of the raw forms."""
    out = []
    for r in records:
        msg = r.getMessage()
        if any(x.lower() in msg.lower() for x in raw):
            out.append(f"{r.name}: {msg}")
    return out


def test_the_commander_log_masks_an_address_he_said(cmd_book, caplog):
    """The send path: the address is said in the sentence itself."""
    with caplog.at_level(logging.INFO):
        res = cmd_book.handle("email the handout to dana at example dot com",
                              source="voice")
        assert "to dana at example dot com" in res.reply, res
        cmd_book.handle("no", source="voice")
        # and said as a CORRECTION at the read-back (the confirm lane's
        # "corrects the draft" line, masked at the logger, not edited)
        cmd_book.handle("email the biosensors handout to Heather Smith",
                        source="voice")
        res = cmd_book.handle("yes, to dana at example dot com", source="voice")
        assert "to dana at example dot com" in res.reply, res
        cmd_book.handle("no", source="voice")
    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert any(ln.startswith("handle ") for ln in lines), lines
    assert _leaks(caplog.records, "dana at example dot com", "dana@example.com") == []
    handled = [ln for ln in lines if ln.startswith("handle ")]
    assert any("d… at example dot com" in ln for ln in handled), handled
    corrected = [ln for ln in lines if ln.startswith("send read-back:")
                 and "corrects the draft" in ln]
    assert corrected and all("d… at example dot com" in ln for ln in corrected), \
        corrected
    assert not FakeSMTP.made


def test_the_commander_log_masks_the_address_he_says_after_a_book_miss(cmd_book,
                                                                       caplog):
    """The book path: Dana is not in the book, Jarvis asks, he says the
    address, and the read-back carries it -- none of that reaches the log
    raw, on any logger."""
    with caplog.at_level(logging.INFO):
        res = cmd_book.handle("email the biosensors handout to Dana", source="voice")
        assert res.reply == "I've no address for Dana, sir. What is it?"
        res = cmd_book.handle("dana at example dot com", source="voice")
        assert "to Dana, at dana at example dot com" in res.reply, res
        cmd_book.handle("no", source="voice")
    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert any(ln.startswith("handle ") for ln in lines), lines
    assert _leaks(caplog.records, "dana at example dot com", "dana@example.com") == []
    assert any("d… at example dot com" in ln for ln in lines
               if ln.startswith("handle ")), lines
    assert not FakeSMTP.made


def test_the_log_mask_covers_args_and_the_format_string_alike(caplog):
    """The filter sits on the commander's logger, so a line written
    tomorrow is covered too -- whichever way the address gets into it."""
    from jarvis import commander as cm
    with caplog.at_level(logging.INFO, logger="jarvis.commander"):
        cm.log.info("probe one: %r", "dana at example dot com")
        cm.log.info("probe two: %s and %s", "dana@example.com", "sam at example dot org")
        cm.log.info("probe three: dana at example dot com in the template")
        cm.log.info("probe four: %(who)s", {"who": "dana at example dot com"})
    assert _leaks(caplog.records, "dana at example dot com", "dana@example.com",
                  "sam at example dot org") == []
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "d… at example dot com" in text and "d…@example.com" in text
    assert "s… at example dot org" in text


# ---- (h) what "Dr Heather" does, pinned for the docs: it is the SAME
#      question as "Heather" only when the one-word row carries that
#      honorific; otherwise it is nobody, and it never picks the Dr row.
def test_dr_heather_is_nobody_unless_the_one_word_row_carries_the_dr(book_file):
    """The three books are written at three SIZES: a same-size rewrite in
    the same filesystem tick has the same stamp and is not re-read (the
    editor's rename-over that a real edit is would be)."""
    write(book_file, [{"name": "Heather", "email": "h@example.com"},
                      {"name": "Heather Jones", "email": "hj@example.com"}])
    book = book_mod.current()
    assert book.resolve("Heather").candidates == ["Heather", "Heather Jones"]
    assert book.resolve("Dr Heather").unknown
    write(book_file, [{"name": "Heather", "email": "h@example.com"},
                      {"name": "Heather Jones", "email": "hj@example.com",
                       "honorific": "Dr", "note": "the Dr is on this row"}])
    assert book.resolve("Dr Heather").unknown, "the Dr on the two-word row is not this"
    assert book.resolve("Dr Heather Jones").addr == "hj@example.com"
    write(book_file, [{"name": "Heather", "email": "h@example.com", "honorific": "Dr"},
                      {"name": "Heather Jones", "email": "hj@example.com"}])
    assert book.resolve("Dr Heather").candidates == ["Heather", "Heather Jones"]
    assert not book.resolve("Dr Heather").found, "never a pick of the Dr row"


# ---- the stamp while broken: a fix that lands in the same tick as the
#      last good write, byte-identical, must still be seen
def test_a_fix_in_the_same_tick_as_the_last_good_write_is_still_seen(two_heathers,
                                                                      monkeypatch):
    """Measured once in a two-file run (test_a_fixed_file_writes_again, 1 in
    326): the fixed file was byte-identical to the one the Book had loaded
    and landed inside the filesystem's ~1 ms tick, so its stamp EQUALLED
    the last good stamp, refresh() short-circuited on it, and the book
    stayed broken with the file already fixed. (Comparing against the
    BROKEN stamp instead fails the other way: a chmod and its undo in one
    tick, test_an_unreadable_file_refuses_a_write.) So a broken book is
    re-read on every refresh. The stamps are held here so the tick is a
    fact of the test, not a race."""
    good = ("mtime-1", 200, "ctime-1")
    held = {"stamp": good}
    monkeypatch.setattr(book_mod, "_stamp", lambda path: held["stamp"])
    book = book_mod.current()
    assert book.broken == "" and book.resolve("Mum").found
    _corrupt(two_heathers)
    held["stamp"] = ("mtime-1", 201, "ctime-1")       # same tick, one byte more
    assert book.add({"name": "Dana Ruiz", "email": "dana@example.com"})[0] is None
    assert book.broken.startswith(BROKEN)
    write(two_heathers, TWO_HEATHERS)
    held["stamp"] = good                    # the fix: identical bytes, same tick
    contact, why = book.add({"name": "Dana Ruiz", "email": "dana@example.com"})
    assert contact is not None and why == "", why
    assert book.broken == ""
    assert book.resolve("Dana Ruiz").addr == "dana@example.com"


def test_the_stamp_carries_the_inode_so_a_rename_over_is_always_a_change(two_heathers):
    """A same-size rewrite that lands in the same tick is missed by
    (mtime_ns, size, ctime_ns); a rename-over (an editor's atomic save,
    the CLI's own write) is a NEW inode, and the inode in the stamp makes
    that half of the window a change every time."""
    before = book_mod._stamp(two_heathers)
    assert two_heathers.stat().st_ino in before, before
    tmp = two_heathers.with_name("saved.json")
    tmp.write_bytes(two_heathers.read_bytes())
    os.replace(tmp, two_heathers)
    after = book_mod._stamp(two_heathers)
    assert after != before
    assert two_heathers.stat().st_ino in after
