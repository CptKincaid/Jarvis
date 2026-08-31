"""People-mail heads-up (jarvis/mailwatch.py): only senders in the people
book announce, each message once, the cap counts the rest, and a box with no
mailbox or no people book never opens a socket. The IMAP layer is replaced
by a callable seam (fetch_unread), the way deadlines.py replaces fetch_due.
"""
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from jarvis.mailwatch import PeopleMailHeadsUp
from jarvis.tools.mail import Mail, MailNotConfigured

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 8, 31, 20, 30, tzinfo=TZ)
CFG = {"gmail": {"address": "hunter@example.com", "app_password": "abcd efgh ijkl"}}
NO_MAIL = {"gmail": {"address": "", "app_password": ""}}
BOOK = {"advisor": {"name": "Dr. Villalobos", "email": "villalobos@tamu.edu"},
        "mom": {"name": "Linda Peyrovi"}}


@pytest.fixture(autouse=True)
def _firewall(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_ASSISTANT_CONFIG", str(tmp_path / "assistant.json"))
    assert not str(os.environ["JARVIS_LOG_DIR"]).startswith("/tmp/vss_voice")
    import urllib.request

    def _no_network(*a, **k):
        raise AssertionError("unit test tried to reach the network")
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)


def _mail(addr, subject, name="", minutes=5, account=""):
    return Mail(from_name=name, from_addr=addr, subject=subject,
                date=NOW - timedelta(minutes=minutes), snippet="body text",
                account=account)


def _make(tmp_path, mails, book=BOOK, cfg=CFG, state="mw.json"):
    said = []
    calls = []

    def fetch(cfg_arg, since_hours=24, limit=20):
        calls.append((since_hours, limit))
        if isinstance(mails, Exception):
            raise mails
        return list(mails)

    w = PeopleMailHeadsUp(cfg, people=lambda: book,
                          announce=lambda title, text: said.append((title, text)),
                          state_path=tmp_path / state, now=lambda: NOW,
                          fetch_unread=fetch)
    return w, said, calls


# ------------------------------------------------------------ the filter
def test_only_people_book_senders_announce(tmp_path):
    mails = [_mail("noreply@linkedin.com", "You appeared in 3 searches"),
             _mail("villalobos@tamu.edu", "thesis draft"),
             _mail("spam@example.com", "URGENT INVOICE")]
    w, said, _c = _make(tmp_path, mails)
    assert w.tick() == 1
    assert said == [("Mail", "Mail from Dr. Villalobos, sir — re: thesis draft.")]


def test_a_people_book_entry_with_only_a_name_matches_the_display_name(tmp_path):
    w, said, _c = _make(tmp_path, [_mail("lp1954@gmail.com", "dinner Sunday?",
                                         name="Linda Peyrovi")])
    assert w.tick() == 1
    assert said[0][1] == "Mail from Linda Peyrovi, sir — re: dinner Sunday?."


def test_a_subjectless_mail_still_says_who_it_is_from(tmp_path):
    w, said, _c = _make(tmp_path, [_mail("villalobos@tamu.edu", "   ")])
    assert w.tick() == 1
    assert said[0][1] == "Mail from Dr. Villalobos, sir."


def test_a_long_subject_is_trimmed(tmp_path):
    subject = "re: " + " ".join(["thesis"] * 40)
    w, said, _c = _make(tmp_path, [_mail("villalobos@tamu.edu", subject)])
    w.tick()
    assert said[0][1].endswith("….") and len(said[0][1]) < 140


# --------------------------------------------------------------- dedupe
def test_each_message_is_announced_once_and_survives_a_restart(tmp_path):
    mails = [_mail("villalobos@tamu.edu", "thesis draft")]
    w, said, _c = _make(tmp_path, mails)
    assert w.tick() == 1
    assert w.tick() == 0, "announced twice"
    w2, said2, _c2 = _make(tmp_path, mails)
    assert w2.tick() == 0 and said2 == []
    assert list(json.loads((tmp_path / "mw.json").read_text()))[0].startswith(
        "|villalobos@tamu.edu|thesis draft|")


def test_the_cap_counts_the_rest_and_marks_them_seen(tmp_path):
    mails = [_mail("villalobos@tamu.edu", f"draft {i}", minutes=i)
             for i in range(1, 7)]
    w, said, _c = _make(tmp_path, mails)
    assert w.tick() == 4                       # 3 lines plus the tally
    assert said[-1][1] == "And 3 more from your contacts, sir."
    assert w.tick() == 0, "the ones held back must not come round again"


def test_a_sink_failure_does_not_repeat_the_line(tmp_path):
    def boom(title, text):
        raise RuntimeError("discord down")

    w = PeopleMailHeadsUp(CFG, people=lambda: BOOK, announce=boom,
                          state_path=tmp_path / "mw.json", now=lambda: NOW,
                          fetch_unread=lambda *a, **k: [
                              _mail("villalobos@tamu.edu", "thesis draft")])
    assert w.tick() == 0
    assert w.tick() == 0


# -------------------------------------------------------------- silence
def test_silent_without_a_mailbox(tmp_path):
    called = []

    def fetch(*a, **k):
        called.append(a)
        raise AssertionError("an unconfigured box must not open a socket")

    w = PeopleMailHeadsUp(NO_MAIL, people=lambda: BOOK, announce=lambda t, x: None,
                          state_path=tmp_path / "mw.json", now=lambda: NOW,
                          fetch_unread=fetch)
    assert w.tick() == 0 and called == []


def test_silent_without_a_people_book(tmp_path):
    calls = []
    w, said, calls = _make(tmp_path, [_mail("villalobos@tamu.edu", "x")], book={})
    assert w.tick() == 0 and said == [] and calls == [], \
        "nobody to watch for: the inbox is not even read"


def test_the_switch_turns_it_off(tmp_path):
    w, said, calls = _make(tmp_path, [_mail("villalobos@tamu.edu", "x")],
                           cfg=dict(CFG, watch={"people_mail": False}))
    assert w.tick() == 0 and calls == []


def test_an_imap_outage_is_swallowed(tmp_path):
    w, said, _c = _make(tmp_path, OSError("connection reset"))
    assert w.tick() == 0 and said == []
    w2, said2, _c2 = _make(tmp_path, MailNotConfigured("gone"))
    assert w2.tick() == 0 and said2 == []


# ---------------------------------------------------------------- thread
def test_the_thread_is_joinable_and_restartable(tmp_path):
    w, _said, _c = _make(tmp_path, [])
    w.start()
    t1 = w._thread
    assert t1 is not None and t1.is_alive()
    w.start()
    assert w._thread is t1, "start is idempotent"
    w.stop()
    assert not t1.is_alive(), "stop() joins the thread out"
    w.start()
    assert w._thread is not t1 and w._thread.is_alive()
    w.stop()
