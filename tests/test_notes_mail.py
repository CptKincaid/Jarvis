"""Notes/to-dos store + tool (spec 6.5) and Gmail IMAP tool (spec 6.6).

Firewall: tmp db, tmp JARVIS_LOG_DIR (conftest) and a tmp
JARVIS_ASSISTANT_CONFIG; no network — the IMAP class is a fake that
returns canned RFC 822 bytes; nothing here touches /tmp/vss_voice.
"""
import email.utils
import imaplib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jarvis.tools import mail as mail_mod
from jarvis.tools import notes as notes_mod
from jarvis.tools.mail import (mail_accounts,Mail, MailNotConfigured, NOTHING_NEW_LINE,
                               UNREACHABLE_LINE, fact_sheet, fetch_unread,
                               imap_date, make_snippet, when_text)
from jarvis.tools.notes import NotesStore, join_spoken, parse_which
from jarvis.tools.registry import ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")

    def _no_network(*a, **k):                    # no real IMAP socket, ever
        raise AssertionError("unit test tried to open a real IMAP connection")
    monkeypatch.setattr(imaplib.IMAP4_SSL, "__init__", _no_network)
    yield


class FakeCfg:
    """Duck-typed AssistantConfig (spec 10.2): get / is_configured / setup_line."""

    def __init__(self, data=None):
        self.data = data or {}

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def is_configured(self, section):
        if section == "gmail":
            g = self.data.get("gmail", {})
            return bool(g.get("address") and g.get("app_password"))
        return False

    def setup_line(self, section):
        return f"I'll need your {section} set up, sir; the notes are in docs/assistant-setup.md."


# ================================================================ notes
@pytest.fixture
def store(tmp_path):
    s = NotesStore(tmp_path / "notes.db")
    yield s
    s.close()


def test_add_list_count_and_persistence(tmp_path):
    s = NotesStore(tmp_path / "n.db")
    a = s.add("note", "  call   the dentist ")
    b = s.add("todo", "buy milk")
    assert (a, b) == (1, 1)
    assert [n["text"] for n in s.list("note")] == ["call the dentist"]
    assert s.count("todo") == 1
    s.close()
    s2 = NotesStore(tmp_path / "n.db")            # survives a re-open
    assert [t["text"] for t in s2.list("todo")] == ["buy milk"]
    s2.close()


def test_add_rejects_bad_kind_and_empty_text(store):
    with pytest.raises(ValueError):
        store.add("recipe", "x")
    with pytest.raises(ValueError):
        store.add("note", "   ")


def test_list_is_chronological_window_of_latest(store):
    for i in range(12):
        store.add("todo", f"task {i}")
    items = store.list("todo", limit=10)
    assert [t["text"] for t in items] == [f"task {i}" for i in range(2, 12)]


def test_list_text_wording(store):
    assert store.list_text("note") == "No notes yet, sir."
    assert store.list_text("todo") == "Nothing on your list, sir."
    store.add("todo", "buy milk")
    assert store.list_text("todo") == "One to-do, sir: buy milk."
    store.add("todo", "call the dentist")
    assert store.list_text("todo") == "Two to-dos, sir: buy milk and call the dentist."
    store.add("todo", "fix the bike.")
    assert store.list_text("todo") == \
        "Three to-dos, sir: buy milk, call the dentist, and fix the bike."
    store.add("note", "the wifi password is on the fridge")
    assert store.list_text("note") == "One note, sir: the wifi password is on the fridge."


def test_list_text_more_than_window(store):
    for i in range(12):
        store.add("note", f"note {i}")
    text = store.list_text("note")
    assert text.startswith("Twelve notes, sir; the latest ten: note 2, note 3")
    assert text.endswith("and note 11.")


def test_list_text_uses_semicolons_when_items_hold_commas(store):
    store.add("todo", "eggs, milk and bread")
    store.add("todo", "call mum")
    store.add("todo", "bike")
    assert store.list_text("todo") == \
        "Three to-dos, sir: eggs, milk and bread; call mum; and bike."


def test_join_spoken_and_number_words():
    assert join_spoken([]) == ""
    assert join_spoken(["a"]) == "a"
    assert join_spoken(["a", "b"]) == "a and b"
    assert join_spoken(["a", "b", "c"]) == "a, b, and c"
    assert notes_mod.number_word(0) == "no"
    assert notes_mod.number_word(3) == "three"
    assert notes_mod.number_word(14) == "fourteen"
    assert notes_mod.number_word(21) == "21"


@pytest.mark.parametrize("which,expected", [
    (None, ("last", None)),
    ("", ("last", None)),
    ("last", ("last", None)),
    ("the last one", ("last", None)),
    ("that one", ("last", None)),
    ("latest", ("last", None)),
    ("all", ("all", None)),
    ("all of them", ("all", None)),
    ("2", ("index", 2)),
    ("#2", ("index", 2)),
    ("number 2", ("index", 2)),
    ("number two", ("index", 2)),
    ("the second one", ("index", 2)),
    ("second", ("index", 2)),
    ("the 3rd", ("index", 3)),
    ("first", ("index", 1)),
    (3, ("index", 3)),
    ("the dentist one", ("text", "dentist")),
    ("Call the Dentist", ("text", "call the dentist")),
])
def test_parse_which(which, expected):
    assert parse_which(which) == expected


def test_resolve_by_ordinal_index_substring_last(store):
    store.add("todo", "buy milk")
    store.add("todo", "call the dentist")
    store.add("todo", "fix the bike")
    assert store.resolve("todo", "last")["text"] == "fix the bike"
    assert store.resolve("todo", "the second one")["text"] == "call the dentist"
    assert store.resolve("todo", "2")["text"] == "call the dentist"
    assert store.resolve("todo", "first")["text"] == "buy milk"
    assert store.resolve("todo", "dentist")["text"] == "call the dentist"
    assert store.resolve("todo", "DENTIST")["text"] == "call the dentist"
    assert store.resolve("todo", "the bike")["text"] == "fix the bike"
    assert store.resolve("todo", "7") is None
    assert store.resolve("todo", "0") is None
    assert store.resolve("todo", "holiday") is None
    assert store.resolve("note", "last") is None          # empty table


def test_resolve_substring_prefers_exact_then_most_recent(store):
    store.add("note", "milk")
    store.add("note", "buy milk tomorrow")
    store.add("note", "milk")
    assert store.resolve("note", "milk")["id"] == 3
    store.add("note", "more milk please")
    assert store.resolve("note", "milk please")["id"] == 4


def test_remove_and_complete(store):
    store.add("todo", "buy milk")
    store.add("todo", "call the dentist")
    store.add("todo", "fix the bike")
    done = store.complete("dentist")
    assert [d["text"] for d in done] == ["call the dentist"] and done[0]["done"] == 1
    assert [t["text"] for t in store.list("todo")] == ["buy milk", "fix the bike"]
    assert len(store.list("todo", include_done=True)) == 3
    assert store.complete("dentist") == []                 # already done
    removed = store.remove("todo", "last")
    assert [r["text"] for r in removed] == ["fix the bike"]
    assert store.remove("todo", "nothing like this") == []
    assert store.count("todo") == 1
    assert len(store.complete("all")) == 1
    # milk done + dentist done; the bike row was deleted outright
    assert store.count("todo") == 0 and store.count("todo", include_done=True) == 2


def test_remove_all_notes(store):
    store.add("note", "a")
    store.add("note", "b")
    assert len(store.remove("note", "all")) == 2
    assert store.count("note") == 0


def test_search(store):
    store.add("note", "The dentist is on Friday")
    store.add("note", "wifi password on the fridge")
    store.add("todo", "book the dentist")
    assert [n["text"] for n in store.search("note", "dentist")] == ["The dentist is on Friday"]
    assert store.search("note", "") == []
    assert store.search_text("note", "dentist") == \
        "One note mentions dentist, sir: The dentist is on Friday."
    assert store.search_text("todo", "plumber") == "Nothing about plumber in your to-dos, sir."
    store.add("note", "dentist bill paid")
    assert store.search_text("note", "dentist") == \
        "Two notes mention dentist, sir: The dentist is on Friday and dentist bill paid."


def test_import_legacy_once(store, tmp_path):
    legacy = tmp_path / "legacy_notes"
    legacy.mkdir()
    (legacy / "note_2026-08-20_09-15-00.txt").write_text(
        "[2026-08-20 09:15]\nRemember to renew the passport\n")
    (legacy / "note_2026-08-21_18-00-00.txt").write_text("no header here\n")
    (legacy / "note_2026-08-22_10-00-00.txt").write_text("[2026-08-22 10:00]\n\n")
    (legacy / "other.txt").write_text("ignored")
    assert store.import_legacy(legacy) == 2
    assert store.import_legacy(legacy) == 0               # marker row
    notes = store.list("note")
    assert [n["text"] for n in notes] == ["Remember to renew the passport", "no header here"]
    assert notes[0]["created"] == datetime(2026, 8, 20, 9, 15).timestamp()
    assert notes[1]["created"] == datetime(2026, 8, 21, 18, 0).timestamp()
    assert notes[0]["tags"] == "legacy:note_2026-08-20_09-15-00.txt"
    assert store.import_legacy(tmp_path / "missing") == 0


def test_notes_tool_confirmations_and_loose_values(store):
    from types import SimpleNamespace
    reg = ToolRegistry()
    reg.register_many(notes_mod.make_tools(FakeCfg(), SimpleNamespace(notes=store)))
    assert reg.names() == ["notes"]
    schema = reg.schemas()[0]["function"]
    assert schema["parameters"]["properties"]["action"]["enum"] == \
        ["add", "list", "remove", "search", "done"]
    assert len(schema["description"].split()) <= 20

    r = reg.call("notes", {"action": "add", "kind": "note", "text": "the car is due a service"})
    assert r.ok and r.speak == "Noted, sir."
    r = reg.call("notes", {"action": "Add", "kind": "to-do", "text": "buy milk", "extra": 1})
    assert r.speak == "Added to your list, sir."
    r = reg.call("notes", {"action": "add", "kind": "tasks", "text": "call the dentist"})
    assert r.speak == "Added to your list, sir."
    r = reg.call("notes", {"action": "add", "kind": "todo", "text": "fix the bike"})
    r = reg.call("notes", {"action": "list", "kind": "todo"})
    assert r.speak == r.text == \
        "Three to-dos, sir: buy milk, call the dentist, and fix the bike."
    r = reg.call("notes", {"action": "done", "kind": "todo", "which": "the second one"})
    assert r.speak == "Done, sir; two left."
    r = reg.call("notes", {"action": "done", "which": "milk"})      # kind defaults to todo
    assert r.speak == "Done, sir; one left."
    r = reg.call("notes", {"action": "done", "kind": "todo", "text": "bike"})
    assert r.speak == "Done, sir; that clears the list."
    r = reg.call("notes", {"action": "done", "kind": "todo", "which": "bike"})
    assert not r.ok and r.speak == "I couldn't find that one on the list, sir."
    r = reg.call("notes", {"action": "search", "kind": "note", "text": "car"})
    assert r.speak == "One note mentions car, sir: the car is due a service."
    r = reg.call("notes", {"action": "remove", "kind": "note", "which": "last"})
    assert r.speak == "Forgotten, sir."
    r = reg.call("notes", {"action": "list", "kind": "note"})
    assert r.speak == "No notes yet, sir."
    r = reg.call("notes", {"action": "add", "kind": "note"})
    assert not r.ok and r.speak == "What shall I note down, sir?"
    r = reg.call("notes", {"action": "add", "kind": "todo", "text": ""})
    assert r.speak == "What shall I add to the list, sir?"
    r = reg.call("notes", {"action": "add", "kind": "todo", "text": "a"})
    r = reg.call("notes", {"action": "add", "kind": "todo", "text": "b"})
    r = reg.call("notes", {"action": "remove", "kind": "todo", "which": "a"})
    assert r.speak == "Struck off, sir; one left."
    r = reg.call("notes", {"action": "remove", "kind": "todo", "which": "b"})
    assert r.speak == "Struck off, sir; the list is clear."
    r = reg.call("notes", {"action": "juggle", "kind": "todo"})
    assert not r.ok


def test_notes_tool_opens_default_store_when_not_wired(tmp_path, monkeypatch):
    from jarvis import config
    monkeypatch.setattr(config.PATHS, "MEMORY_DIR", tmp_path / "mem")
    monkeypatch.setattr(config.PATHS, "NOTES_DB", tmp_path / "mem" / "notes.db",
                        raising=False)
    (spec,) = notes_mod.make_tools(None, None)
    r = spec.handler(action="add", kind="note", text="hello")
    assert r.speak == "Noted, sir."
    assert (tmp_path / "mem" / "notes.db").exists()


# ================================================================= mail
def _hdr(fields: dict) -> bytes:
    return "".join(f"{k}: {v}\r\n" for k, v in fields.items()).encode()


def _fetch_data(messages):
    """imaplib-shaped FETCH payload for [(headers, body), …]."""
    data = []
    for i, (h, b) in enumerate(messages, 1):
        data.append((f"{i} (BODY[HEADER.FIELDS (FROM SUBJECT DATE CONTENT-TYPE "
                     f"CONTENT-TRANSFER-ENCODING)] {{{len(h)}}}".encode(), h))
        data.append((f" BODY[TEXT]<0> {{{len(b)}}}".encode(), b))
        data.append(b")")
    return data


def _canned(now):
    """Four messages: RFC 2047 headers + QP body with quotes/signature,
    multipart/alternative, html-only, and one older than the window."""
    d = email.utils.format_datetime
    m1 = (_hdr({"From": "=?UTF-8?Q?Jos=C3=A9_M=C3=BCller?= <jose@example.com>",
                "Subject": "=?utf-8?b?SW52b2ljZSA0NDcxIGR1ZSBGcmlkYXk=?=",
                "Date": d(now.replace(hour=14, minute=10)),
                "Content-Type": "text/plain; charset=utf-8",
                "Content-Transfer-Encoding": "quoted-printable"}),
          b"Hi Hunter,\r\nThe invoice for the caf=C3=A9 is attached; it is due Friday.\r\n"
          b"> earlier quoted text\r\n> more quoted\r\nOn Mon, Aug 24, 2026 Jane wrote:\r\n"
          b"old stuff\r\n-- \r\nJose\r\n")
    boundary = "b1"
    m2 = (_hdr({"From": "Jane Doe <jane@example.com>",
                "Subject": "Standup moved",
                "Date": d(now - timedelta(hours=3)),
                "Content-Type": f'multipart/alternative; boundary="{boundary}"'}),
          f"--{boundary}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
          f"Standup is at 2:30 pm today.\r\n\r\nBest regards,\r\nJane\r\n"
          f"--{boundary}\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
          f"<p>Standup is at <b>2:30 pm</b> today.</p>\r\n--{boundary}--\r\n".encode())
    m3 = (_hdr({"From": "newsletter@shop.example",
                "Subject": "Sale ends tonight",
                "Date": d(now - timedelta(hours=20)),
                "Content-Type": "text/html; charset=utf-8"}),
          b"<html><head><style>p{}</style></head><body><h1>Big sale</h1>"
          b"<p>Everything 20% off &amp; free shipping.</p></body></html>")
    m4 = (_hdr({"From": "Old <old@example.com>", "Subject": "Ancient",
                "Date": d(now - timedelta(days=3)),
                "Content-Type": "text/plain"}),
          b"too old to count\r\n")
    return [m1, m2, m3, m4]


class FakeIMAP:
    """Records the IMAP conversation; returns canned data."""
    instances: list = []
    messages: list = []
    fail_login = False
    fail_connect = False

    def __init__(self, host, port=993, timeout=None):
        if self.fail_connect:
            raise OSError("connection refused")
        self.host, self.port, self.timeout = host, port, timeout
        self.calls = []
        FakeIMAP.instances.append(self)

    def login(self, user, password):
        self.calls.append(("login", user))
        if self.fail_login:
            raise imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")
        return "OK", [b"ok"]

    def select(self, mailbox="INBOX", readonly=False):
        self.calls.append(("select", mailbox, readonly))
        return "OK", [b"4"]

    def search(self, charset, *criteria):
        self.calls.append(("search", charset) + criteria)
        n = len(self.messages)
        return "OK", [" ".join(str(i) for i in range(1, n + 1)).encode() if n else b""]

    def fetch(self, ids, spec):
        self.calls.append(("fetch", ids, spec))
        return "OK", _fetch_data(self.messages)

    def logout(self):
        self.calls.append(("logout",))
        return "BYE", [b"bye"]


@pytest.fixture
def fake_imap(monkeypatch):
    FakeIMAP.instances = []
    FakeIMAP.messages = []
    monkeypatch.setattr(FakeIMAP, "fail_login", False)
    monkeypatch.setattr(FakeIMAP, "fail_connect", False)
    return FakeIMAP


GMAIL_CFG = {"gmail": {"address": "hunter@example.com",
                       "app_password": "abcd efgh ijkl mnop",
                       "imap_host": "imap.gmail.com"}}
NOW = datetime(2026, 8, 26, 16, 0).astimezone()


def test_imap_date_is_english_regardless_of_locale():
    assert imap_date(datetime(2026, 8, 25)) == "25-Aug-2026"
    assert imap_date(datetime(2026, 1, 5)) == "05-Jan-2026"


def test_fetch_unread_conversation_and_decoding(fake_imap, caplog):
    fake_imap.messages = _canned(NOW)
    with caplog.at_level(logging.DEBUG, logger="jarvis"):
        mails = fetch_unread(FakeCfg(GMAIL_CFG), since_hours=24, limit=20,
                             imap=fake_imap, now=NOW)
    conn = fake_imap.instances[0]
    assert (conn.host, conn.port) == ("imap.gmail.com", 993) and conn.timeout
    kinds = [c[0] for c in conn.calls]
    assert kinds == ["login", "select", "search", "fetch", "logout"]
    assert conn.calls[1] == ("select", "INBOX", True)          # readonly
    assert conn.calls[2] == ("search", None, "UNSEEN", "SINCE 25-Aug-2026")
    fetch_spec = conn.calls[3][2]
    assert "BODY.PEEK[" in fetch_spec and "RFC822" not in fetch_spec
    assert "BODY[TEXT]" not in fetch_spec and "BODY[HEADER" not in fetch_spec
    assert f"<0.{mail_mod.BODY_BYTES}>" in fetch_spec
    assert conn.calls[3][1] == b"1,2,3,4"

    assert [m.subject for m in mails] == \
        ["Invoice 4471 due Friday", "Standup moved", "Sale ends tonight"]
    m1, m2, m3 = mails
    assert (m1.from_name, m1.from_addr) == ("José Müller", "jose@example.com")
    assert m1.snippet == "Hi Hunter, The invoice for the café is attached; it is due Friday."
    assert m1.date.hour == 14 and m1.date.minute == 10
    assert m2.sender == "Jane Doe"
    assert m2.snippet == "Standup is at 2:30 pm today."
    assert m3.sender == "newsletter@shop.example"
    assert m3.snippet == "Big sale Everything 20% off & free shipping."
    # password and bodies never reach the log
    text = caplog.text
    assert "abcd efgh" not in text and "invoice" not in text.lower()
    assert "h…@example.com" in text
    log_file = Path(os.environ["JARVIS_LOG_DIR"]) / "jarvis.log"
    if log_file.exists():
        assert "abcd efgh" not in log_file.read_text()


def test_fetch_unread_limit_and_empty(fake_imap):
    fake_imap.messages = _canned(NOW)
    fetch_unread(FakeCfg(GMAIL_CFG), limit=2, imap=fake_imap, now=NOW)
    assert fake_imap.instances[-1].calls[3][1] == b"3,4"       # newest ids
    fake_imap.messages = []
    assert fetch_unread(FakeCfg(GMAIL_CFG), imap=fake_imap, now=NOW) == []
    assert [c[0] for c in fake_imap.instances[-1].calls] == \
        ["login", "select", "search", "logout"]


def test_fetch_unread_not_configured(fake_imap):
    with pytest.raises(MailNotConfigured):
        fetch_unread(FakeCfg({"gmail": {"address": "", "app_password": ""}}), imap=fake_imap)
    with pytest.raises(MailNotConfigured):
        fetch_unread({"gmail": {"address": "<your address>", "app_password": "x"}},
                     imap=fake_imap)
    with pytest.raises(MailNotConfigured):
        fetch_unread(None, imap=fake_imap)
    assert fake_imap.instances == []


def test_fetch_unread_with_plain_dict_config(fake_imap):
    fake_imap.messages = _canned(NOW)[:1]
    mails = fetch_unread(dict(GMAIL_CFG), imap=fake_imap, now=NOW)
    assert len(mails) == 1 and fake_imap.instances[0].host == "imap.gmail.com"


def test_make_snippet_rules():
    assert make_snippet("line one\n> quoted\nline two\n-- \nsig") == "line one line two"
    assert make_snippet("hello\nOn Tue, 25 Aug 2026 at 10:00, Bob <b@x> wrote:\n> hi") == "hello"
    assert make_snippet("a\nSent from my iPhone\nb") == "a"
    long = " ".join(["word"] * 80)
    s = make_snippet(long)
    assert len(s) <= 201 and s.endswith("…")
    assert make_snippet("") == ""


def test_when_text_and_fact_sheet():
    now = NOW
    today = now.replace(hour=14, minute=10)
    assert when_text(today, now) == "2:10 pm"
    assert when_text(now - timedelta(days=1), now) == "yesterday 4:00 pm"
    assert when_text(now - timedelta(days=3), now) == "Sun 4:00 pm"
    assert when_text(None, now) == ""
    mails = [Mail("Jane Doe", "jane@x", "Invoice 4471 due Friday", today, "Please pay soon"),
             Mail("", "bot@x", "Build failed", None, "")]
    sheet = fact_sheet(mails, total=5, since_hours=24, now=now)
    assert sheet == ("5 unread since yesterday, latest 2:\n"
                     "1) Jane Doe — Invoice 4471 due Friday (2:10 pm): Please pay soon\n"
                     "2) bot@x — Build failed")
    assert fact_sheet(mails, total=2, now=now).startswith("2 unread since yesterday:\n")


def test_get_mail_tool(fake_imap):
    from types import SimpleNamespace
    fake_imap.messages = _canned(datetime.now().astimezone())
    reg = ToolRegistry()
    reg.register_many(mail_mod.make_tools(FakeCfg(GMAIL_CFG), SimpleNamespace(imap=fake_imap)))
    assert reg.names() == ["get_mail"]
    assert len(reg.schemas()[0]["function"]["description"].split()) <= 20
    r = reg.call("get_mail", {"limit": "2", "noise": True})
    assert r.ok and r.max_sentences == 4 and r.speak is None
    lines = r.text.splitlines()
    assert lines[0] == "3 unread since yesterday, latest 2:"
    # newest two of the three (which is newest depends on the wall clock)
    heads = {ln.split(" — ")[0][3:] for ln in lines[1:]}
    assert heads == {"José Müller", "Jane Doe"}
    assert lines[1].startswith("1) ") and lines[2].startswith("2) ")
    assert all(" (" in ln and "): " in ln for ln in lines[1:])
    r = reg.call("get_mail", {})
    assert r.text.splitlines()[0] == "3 unread since yesterday:"
    assert r.text.count("\n") == 3

    fake_imap.messages = []
    r = reg.call("get_mail", {"limit": 5})
    assert r.ok and r.speak == NOTHING_NEW_LINE

    fake_imap.fail_login = True
    r = reg.call("get_mail", {})
    assert not r.ok and r.speak == UNREACHABLE_LINE and "unreachable" in r.text
    fake_imap.fail_login = False
    fake_imap.fail_connect = True
    r = reg.call("get_mail", {})
    assert not r.ok and r.speak == UNREACHABLE_LINE


def test_get_mail_tool_unconfigured(fake_imap):
    from types import SimpleNamespace
    cfg = FakeCfg({"gmail": {"address": "", "app_password": ""}})
    (spec,) = mail_mod.make_tools(cfg, SimpleNamespace(imap=fake_imap))
    r = spec.handler()
    assert not r.ok
    assert r.speak == r.text == \
        "I'll need your gmail set up, sir; the notes are in docs/assistant-setup.md."
    assert fake_imap.instances == []
    # no config object at all -> the module's own persona fallback
    (spec,) = mail_mod.make_tools(None, None)
    r = spec.handler(limit="lots")
    assert not r.ok and r.speak == mail_mod.FALLBACK_SETUP_LINE


def test_parse_fetch_tolerates_odd_shapes():
    assert mail_mod._parse_fetch(None) == []
    assert mail_mod._parse_fetch([b")", "junk"]) == []
    data = [(b"1 (BODY[TEXT]<0> {3}", b"abc"), b")",
            (b"2 (BODY[HEADER.FIELDS (FROM)] {9}", b"From: a\r\n"), b")"]
    assert mail_mod._parse_fetch(data) == [(b"", b"abc"), (b"From: a\r\n", b"")]


def test_tool_result_shape():
    r = ToolResult(text="x")
    assert r.ok and r.max_sentences == 2 and r.card is None and r.speak is None


# ============================================ mail through the tool loop
def test_get_mail_summarised_by_the_local_model(fake_imap, monkeypatch):
    """Spec 6.6: the fact sheet is handed to the local model, which speaks
    a persona summary. The model is mocked — no Ollama, no network."""
    from types import SimpleNamespace

    from jarvis import brain as brain_mod

    fake_imap.messages = _canned(datetime.now().astimezone())
    reg = ToolRegistry()
    reg.register_many(mail_mod.make_tools(FakeCfg(GMAIL_CFG),
                                          SimpleNamespace(imap=fake_imap)))
    sent = []

    def fake_http(path, payload=None, timeout=None):
        assert path == "/api/chat"
        sent.append(payload)
        if len(sent) == 1:
            assert [t["function"]["name"] for t in payload["tools"]] == ["get_mail"]
            return {"message": {"content": "",
                                "tool_calls": [{"function": {
                                    "name": "get_mail",
                                    "arguments": '{"limit": "3"}'}}]}}
        return {"message": {"content": "Three unread, sir: an invoice from "
                                       "Jane Doe, a build failure, and a note "
                                       "from Jose Muller."}}

    monkeypatch.setattr(brain_mod, "_http", fake_http)
    brain = brain_mod.JarvisBrain(None, None, registry=reg)
    tags = brain._chat_sync("any mail?")

    assert len(sent) == 2, "the fact sheet must go back to the model"
    tool_turn = sent[1]["messages"][-1]
    assert tool_turn["role"] == "tool" and tool_turn["tool_name"] == "get_mail"
    assert tool_turn["content"].startswith("3 unread since yesterday:")
    assert "Invoice 4471" in tool_turn["content"]
    assert tags == [("SPEAK", "Three unread, sir: an invoice from Jane Doe, "
                              "a build failure, and a note from Jose Muller.")]


def test_get_mail_excuse_is_spoken_without_a_model_turn(fake_imap, monkeypatch):
    """A ToolResult.speak (nothing new / not set up) short-circuits the
    loop: the excuse is spoken verbatim, the model renders nothing."""
    from types import SimpleNamespace

    from jarvis import brain as brain_mod

    fake_imap.messages = []
    reg = ToolRegistry()
    reg.register_many(mail_mod.make_tools(FakeCfg(GMAIL_CFG),
                                          SimpleNamespace(imap=fake_imap)))
    calls = []

    def fake_http(path, payload=None, timeout=None):
        calls.append(payload)
        return {"message": {"content": "",
                            "tool_calls": [{"function": {"name": "get_mail",
                                                         "arguments": {}}}]}}

    monkeypatch.setattr(brain_mod, "_http", fake_http)
    tags = brain_mod.JarvisBrain(None, None, registry=reg)._chat_sync("mail?")
    assert tags == [("SPEAK", NOTHING_NEW_LINE)]
    assert len(calls) == 1, "no second model turn once the tool has spoken"


# ------------------------------------------------- multiple mailboxes
#
# `gmail` held exactly one address/app_password pair. Hunter runs three
# accounts (personal, work, school), so a single slot meant two of them were
# invisible to the briefing. `gmail.accounts` is a list; the legacy top-level
# pair still works so existing configs keep running untouched.
MULTI_CFG = {"gmail": {"accounts": [
    {"label": "personal", "address": "me@gmail.com", "app_password": "aaaa bbbb cccc dddd"},
    {"label": "work", "address": "me@work.com", "app_password": "eeee ffff gggg hhhh"},
]}}


def test_legacy_single_account_config_still_works(fake_imap):
    accts = mail_accounts(FakeCfg(GMAIL_CFG))
    assert [a["address"] for a in accts] == ["hunter@example.com"]
    assert accts[0]["host"] == "imap.gmail.com"


def test_accounts_list_is_used_when_present():
    accts = mail_accounts(FakeCfg(MULTI_CFG))
    assert [a["address"] for a in accts] == ["me@gmail.com", "me@work.com"]
    assert [a["label"] for a in accts] == ["personal", "work"]


def test_unlabelled_accounts_get_a_label_from_the_address():
    cfg = {"gmail": {"accounts": [
        {"address": "hunterpey@school.edu", "app_password": "aaaa bbbb cccc dddd"}]}}
    assert mail_accounts(FakeCfg(cfg))[0]["label"] == "hunterpey"


def test_incomplete_accounts_are_skipped_not_fatal():
    cfg = {"gmail": {"accounts": [
        {"address": "good@gmail.com", "app_password": "aaaa bbbb cccc dddd"},
        {"address": "", "app_password": "x"},
        {"address": "<your address>", "app_password": "y"},
        {"address": "nopassword@gmail.com", "app_password": ""},
    ]}}
    assert [a["address"] for a in mail_accounts(FakeCfg(cfg))] == ["good@gmail.com"]


def test_fetch_unread_reads_every_account_and_tags_each_mail(fake_imap):
    fake_imap.messages = _canned(NOW)[:2]
    mails = fetch_unread(FakeCfg(MULTI_CFG), imap=fake_imap, now=NOW)

    logins = [c[1] for i in fake_imap.instances for c in i.calls if c[0] == "login"]
    assert logins == ["me@gmail.com", "me@work.com"], "did not visit both mailboxes"
    assert len(mails) == 4, "both mailboxes' mail should be merged"
    assert {m.account for m in mails} == {"personal", "work"}


def test_merged_mail_is_newest_first_across_accounts(fake_imap):
    fake_imap.messages = _canned(NOW)[:2]
    mails = fetch_unread(FakeCfg(MULTI_CFG), imap=fake_imap, now=NOW)
    dates = [m.date for m in mails if m.date]
    assert dates == sorted(dates, reverse=True), "merge did not re-sort"


def test_one_bad_mailbox_does_not_lose_the_others(fake_imap, monkeypatch):
    """A wrong app password on one account must not blind Jarvis to the rest."""
    fake_imap.messages = _canned(NOW)[:1]
    real_login = fake_imap.login

    def selective_login(self, user, password):
        if user == "me@gmail.com":
            self.calls.append(("login", user))
            raise imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")
        return real_login(self, user, password)

    monkeypatch.setattr(fake_imap, "login", selective_login)
    mails = fetch_unread(FakeCfg(MULTI_CFG), imap=fake_imap, now=NOW)

    assert mails, "the healthy mailbox was lost with the broken one"
    assert {m.account for m in mails} == {"work"}


def test_every_mailbox_failing_still_raises(fake_imap):
    fake_imap.fail_login = True
    with pytest.raises(imaplib.IMAP4.error):
        fetch_unread(FakeCfg(MULTI_CFG), imap=fake_imap, now=NOW)


def test_no_accounts_at_all_is_not_configured(fake_imap):
    with pytest.raises(MailNotConfigured):
        fetch_unread(FakeCfg({"gmail": {"accounts": []}}), imap=fake_imap, now=NOW)
