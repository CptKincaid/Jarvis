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
