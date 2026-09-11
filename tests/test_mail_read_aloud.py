"""Reading ONE email aloud — his ask of 2026-09-11.

He reported: "when asked to talk about email he talks about the subject
then doesnt work reading the rest and locks me out". That was exact and
literal. A listing fetch asked IMAP for 2000 bytes of body and kept 200
characters of it, so sender and subject really was all Jarvis had; "read
the rest" matched no command, and "read it" read his X selection instead.

Asked whether he wanted bodies read aloud at all, he said yes. This file
pins the fetching half. The speaking half is in tests/test_commander.py.

No mailbox is touched: every test drives the injected `imap` seam with a
fake, exactly as tests/test_notes_mail.py does.
"""
from __future__ import annotations

import email.utils
from datetime import datetime, timedelta

from jarvis.tools import mail as M
from tests.test_notes_mail import FakeCfg, FakeIMAP, _fetch_data, fake_imap  # noqa: F401

GMAIL_CFG = {"gmail": {"address": "hunter@example.com",
                       "app_password": "abcd efgh ijkl mnop",
                       "imap_host": "imap.gmail.com"}}
NOW = datetime(2026, 9, 11, 18, 0).astimezone()

# A body far longer than a listing's 200-character preview, so "is this a
# body or a preview" is a measurement rather than an opinion.
LONG = ("Dear Hunter, " + "the quarterly figures are attached and the "
        "committee would like a note by Friday. " * 40)


def _msg(subject, sender, body, when):
    hdr = (f"From: {sender}\r\n"
           f"Subject: {subject}\r\n"
           f"Date: {email.utils.format_datetime(when)}\r\n"
           "Content-Type: text/plain; charset=utf-8\r\n").encode()
    return hdr, body.encode()


def _two(now):
    return [_msg("Quarterly figures", "Ali <ali@example.com>", LONG,
                 now - timedelta(hours=2)),
            _msg("Lunch", "Heather <h@example.com>", "Tomorrow at one?",
                 now - timedelta(hours=5))]


def _spec(conn):
    return [c for c in conn.calls if c[0] == "fetch"][0][2]


def test_a_listing_still_asks_for_only_a_preview(fake_imap):
    """The miserly numbers are the RIGHT ones for a browse -- three
    mailboxes, twenty messages. Reading aloud must not make them worse."""
    fake_imap.messages = _two(NOW)
    M.fetch_unread(FakeCfg(GMAIL_CFG), imap=fake_imap, now=NOW,
                   unread_only=False)
    assert f"<0.{M.BODY_BYTES}>" in _spec(fake_imap.instances[0])
    assert M.BODY_BYTES == 2000


def test_reading_one_asks_for_a_whole_message_instead(fake_imap):
    fake_imap.messages = _two(NOW)
    M.read_one(FakeCfg(GMAIL_CFG), imap=fake_imap, now=NOW)
    assert f"<0.{M.READ_BODY_BYTES}>" in _spec(fake_imap.instances[0])
    assert M.READ_BODY_BYTES > M.BODY_BYTES


def test_the_body_that_comes_back_is_a_BODY_not_a_preview(fake_imap):
    """HIS BUG in one assertion. Under the listing caps there was nothing
    to read: 200 characters is a subject line with ambitions."""
    fake_imap.messages = _two(NOW)
    preview = M.fetch_unread(FakeCfg(GMAIL_CFG), imap=fake_imap, now=NOW,
                             unread_only=False)[0].snippet
    fake_imap.messages = _two(NOW)
    full = M.read_one(FakeCfg(GMAIL_CFG), imap=fake_imap, now=NOW).snippet
    assert len(preview) <= M.SNIPPET_CHARS + 1          # +1 for the ellipsis
    assert len(full) > 10 * len(preview)


def test_the_newest_message_is_the_one_read(fake_imap):
    fake_imap.messages = _two(NOW)
    got = M.read_one(FakeCfg(GMAIL_CFG), imap=fake_imap, now=NOW)
    assert got.subject == "Quarterly figures"


def test_a_subject_narrows_it_and_a_miss_is_None_not_the_wrong_mail(fake_imap):
    """A wrong email read aloud IN FULL is worse than no email at all."""
    fake_imap.messages = _two(NOW)
    assert M.read_one(FakeCfg(GMAIL_CFG), subject="lunch", imap=fake_imap,
                      now=NOW).subject == "Lunch"
    fake_imap.messages = _two(NOW)
    assert M.read_one(FakeCfg(GMAIL_CFG), subject="mortgage", imap=fake_imap,
                      now=NOW) is None


def test_a_sender_narrows_it_too(fake_imap):
    fake_imap.messages = _two(NOW)
    got = M.read_one(FakeCfg(GMAIL_CFG), sender="Heather", imap=fake_imap,
                     now=NOW)
    assert got is not None and got.subject == "Lunch"


def test_read_mail_counts_because_he_was_just_told_about_it(fake_imap):
    """A message Jarvis has just described may well have been read
    already; searching UNSEEN for it would answer "there is no such mail"
    about the one it mentioned ten seconds ago."""
    fake_imap.messages = _two(NOW)
    M.read_one(FakeCfg(GMAIL_CFG), imap=fake_imap, now=NOW)
    criteria = [c for c in fake_imap.instances[0].calls if c[0] == "search"][0]
    assert "UNSEEN" not in criteria


def test_the_spoken_line_names_the_message_before_reading_it(fake_imap):
    """By the time he says "read it" he may have been told about three."""
    fake_imap.messages = _two(NOW)
    line = M.spoken_body(M.read_one(FakeCfg(GMAIL_CFG), subject="lunch",
                                    imap=fake_imap, now=NOW))
    assert line.startswith("From Heather")
    assert "subject: Lunch" in line
    assert line.rstrip().endswith("Tomorrow at one?")


def test_nothing_to_read_is_an_empty_string_never_a_crash():
    assert M.spoken_body(None) == ""


# ==================================================================
# THE SPOKEN HALF (jarvis/commander.py)
# ==================================================================
import time                                              # noqa: E402
from types import SimpleNamespace                        # noqa: E402

import jarvis.commander as C                             # noqa: E402


class Reader:
    def __init__(self):
        self.read = []

    def read_text(self, text, label=""):
        self.read.append((text, label))
        return SimpleNamespace(ok=True, message="Reading.")


def _cmdr(last_mail=None, last_document=None, reader=None):
    c = object.__new__(C.Commander)
    svc = SimpleNamespace(assistant=FakeCfg(GMAIL_CFG), reader=reader or Reader())
    if last_mail is not None:
        svc.last_mail = last_mail
    c.services = svc
    c._raw_text = ""
    c._last_document = last_document
    c._svc = lambda n: getattr(svc, n, None)
    c._bg = lambda fn: fn()                       # run it here, not on a thread
    return c, svc


def _mail_ref(at=None, subject="Lunch", sender="Heather"):
    return {"sender": sender, "subject": subject, "account": "",
            "at": time.time() if at is None else at}


def test_read_the_rest_is_a_command_at_all(fake_imap, monkeypatch):
    """The literal sentence from his report. It matched NOTHING before:
    not the selection branch, not the document branch, not inline text."""
    assert C.read_mail_kind("read the rest")
    assert C.read_mail_kind("read that email")
    assert not C.read_mail_kind("read the document")


def test_reading_it_hands_the_BODY_to_the_reader(fake_imap, monkeypatch):
    fake_imap.messages = _two(NOW)
    c, svc = _cmdr(last_mail=_mail_ref())
    svc.imap = fake_imap                      # the injected seam, no patching
    res = C._h_read_mail(c, "read the rest", None)
    assert res.handled and res.speak
    assert svc.reader.read, "the reader was never given the message"
    body, label = svc.reader.read[0]
    assert body.startswith("From Heather")
    assert "Tomorrow at one?" in body


def test_reading_it_does_not_re_open_the_microphone():
    """His ruling the same day: an action ends the turn. A live mic
    through a minute of email read aloud is the worst case available."""
    c, svc = _cmdr(last_mail=_mail_ref())
    c._svc = lambda n: getattr(svc, n, None)
    C._h_read_mail(c, "read the rest", None)
    assert svc.no_followup is True


def test_with_no_message_in_hand_it_says_so_rather_than_guessing():
    c, svc = _cmdr()
    res = C._h_read_mail(c, "read the rest", None)
    assert res.reply == C.NO_MAIL_TO_READ_LINE
    assert not svc.reader.read


def test_a_message_older_than_the_window_is_not_in_hand():
    c, svc = _cmdr(last_mail=_mail_ref(at=time.time() - C.LAST_MAIL_S - 1))
    assert C._h_read_mail(c, "read the rest", None).reply == C.NO_MAIL_TO_READ_LINE


def test_READ_IT_goes_to_whichever_he_was_shown_MOST_RECENTLY():
    """Three claimants for "it" now -- selection, document, message -- and
    the rule between them is recency, never a fixed priority order. A
    fixed order would be wrong half the time by construction."""
    now = time.time()
    # the message came second: "read it" means the message
    c, svc = _cmdr(last_mail=_mail_ref(at=now),
                   last_document=("/tmp/paper.pdf", now - 60))
    c._raw_text = "read it"
    res = _h_read(c)
    assert svc.no_followup is True, "the document won a race it should lose"
    # the document came second: "read it" means the document
    c, svc = _cmdr(last_mail=_mail_ref(at=now - 60),
                   last_document=("/tmp/paper.pdf", now))
    c._raw_text = "read it"
    svc.reader.read_document = lambda path: SimpleNamespace(ok=True,
                                                            message="Reading.")
    _h_read(c)
    assert getattr(svc, "no_followup", False) is False


def _h_read(c):
    return C._h_read_aloud(c, "read it", ("selection", None))
