"""Email a file, by voice (2026-09-02): jarvis/filephrase.py, jarvis/outbox.py
and the commander's `send file` family.

The feature is irreversible, so most of this file is about the ways it must
REFUSE or ASK rather than the one way it sends. Four groups:

* resolving a spoken description to one path -- two similar names, a
  half-remembered name, no match, several matches, a symlink out of the
  roots, an explicit path, a folder, an empty file, an oversized one;
* the recipient and the account -- the people book, the contacts map, an
  address said aloud, an unknown name, three identities and no hint;
* the read-back itself -- what it names, and that a stray "Yeah, so you
  should be able to look that up." cannot answer it;
* the send, driven end to end through a FAKE SMTP class. Nothing here opens
  a socket; tests/conftest.py refuses ports 25/465/587 outright, with no
  environment escape, so a test that lost its fake fails loudly instead of
  emailing one of his correspondents.
"""
import logging
import re
import smtplib
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis import filephrase, outbox
from jarvis.commander import (Commander, IntentClassifier, REGISTRY,
                              ASSISTANT_TIER1, _SEND_FILE_RX, _SEND_NOT_RX,
                              parse_send_answer)
from jarvis.config import CONFIG
from jarvis.tools import filepick
from jarvis.tools import mail as mail_mod


# ------------------------------------------------------------- fixtures
@pytest.fixture
def home(tmp_path):
    """A Desktop / Downloads / Documents with known files and mtimes."""
    for name in ("Desktop", "Downloads", "Documents"):
        (tmp_path / name).mkdir()
    files = {
        "Desktop/lab_report.pdf": b"L" * 3000,
        "Desktop/lab_report_final.pdf": b"F" * 4000,
        "Desktop/notes.txt": b"N" * 800,
        "Downloads/invoice-4471.pdf": b"I" * 2000,
        "Documents/Biosensors_Lab-Handout_v2.pdf": b"B" * 5000,
    }
    now = time.time()
    for i, (rel, data) in enumerate(files.items()):
        p = tmp_path / rel
        p.write_bytes(data)
        # Spread the mtimes an hour apart so "the newest" is unambiguous
        # where a test wants it to be, and TIE_WINDOW_S is not the thing
        # under test unless it says so.
        import os
        os.utime(p, (now - i * 3600, now - i * 3600))
    return tmp_path


@pytest.fixture
def roots(home):
    return [str(home / "Desktop"), str(home / "Downloads"),
            str(home / "Documents")]


class Cfg:
    """The slice of AssistantConfig this feature reads."""

    def __init__(self, **over):
        self.data = {
            "gmail.accounts": [
                {"label": "personal", "address": "hunter@gmail.com",
                 "app_password": "personal-secret"},
                {"label": "work", "address": "hunter@work.com",
                 "app_password": "work-secret"},
                {"label": "school", "address": "hp@tamu.edu",
                 "app_password": "school-secret"},
            ],
            "send_file.contacts": {"heather": "heather@example.com"},
        }
        self.data.update(over)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setup_line(self, section):
        return f"I'll need {section} set up, sir."


class FakeSMTP:
    """Stands in for smtplib.SMTP_SSL. Records instead of connecting."""
    made = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.logged_in = None
        self.sent = []
        self.quit_called = False
        FakeSMTP.made.append(self)

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        self.quit_called = True


@pytest.fixture(autouse=True)
def _fresh_fakes():
    FakeSMTP.made = []
    yield
    FakeSMTP.made = []


def cfg_with_roots(roots, **over):
    return Cfg(**{"send_file.roots": roots, **over})


# ==================================================================
# 1. Resolving a spoken description to ONE path
# ==================================================================
def test_two_similar_names_ask_instead_of_picking(roots):
    """The brief's first hard case. lab_report.pdf and lab_report_final.pdf
    both answer to "the lab report"; picking either silently is how the
    wrong file is attached."""
    m = filephrase.resolve("the lab report", roots=roots)
    assert m.ambiguous and not m.ok
    assert {p.name for p in m.candidates} == {"lab_report.pdf",
                                              "lab_report_final.pdf"}


def test_names_that_score_alike_are_a_question_and_never_a_ranking(roots):
    """Both files answer to "lab report" equally well, and this lane's tie
    band is deliberately wider than filepick's so a near-miss is a question
    too. Pinned here because the constant is the whole safety margin."""
    a = filepick.score("lab report", "lab_report.pdf")
    b = filepick.score("lab report", "lab_report_final.pdf")
    assert abs(a - b) <= filephrase.TIE_SCORE
    assert filephrase.TIE_SCORE > 0.02


def test_a_half_remembered_name_still_resolves(roots):
    """"the biosensors handout" -> Biosensors_Lab-Handout_v2.pdf."""
    m = filephrase.resolve("the biosensors handout", roots=roots)
    assert m.ok and m.path.name == "Biosensors_Lab-Handout_v2.pdf"


def test_a_name_that_matches_nothing_is_not_found(roots):
    m = filephrase.resolve("the widget specification", roots=roots)
    assert not m.ok and not m.ambiguous and m.reason == "not-found"


def test_an_exact_filename_beats_a_near_neighbour(roots):
    """"lab_report.pdf" is not half-remembered; lab_report_final.pdf must
    not tie with it. This is also the regression for "pdf" being eaten as
    a TYPE cue out of the middle of the name he said."""
    m = filephrase.resolve("lab_report.pdf", roots=roots)
    assert m.ok and m.path.name == "lab_report.pdf"
    assert filephrase.parse("lab_report.pdf").name == "lab_report.pdf"


def test_that_file_on_my_desktop_asks_when_the_desktop_has_several(roots):
    m = filephrase.resolve("that file on my desktop", roots=roots)
    assert m.ambiguous
    assert all(p.parent.name == "Desktop" for p in m.candidates)


def test_that_file_on_my_desktop_resolves_when_there_is_only_one(tmp_path):
    (tmp_path / "Desktop").mkdir()
    only = tmp_path / "Desktop" / "one.pdf"
    only.write_bytes(b"x" * 10)
    m = filephrase.resolve("that file on my desktop",
                           roots=[str(tmp_path / "Desktop")])
    assert m.ok and m.path == only.resolve()


def test_the_pdf_i_just_downloaded_uses_type_folder_and_recency(roots):
    m = filephrase.resolve("the PDF I just downloaded", roots=roots)
    assert m.ok and m.path.name == "invoice-4471.pdf"


def test_just_downloaded_asks_when_the_newest_is_old(roots, home):
    """"just" is a claim about WHEN. A file from last week is not it."""
    import os
    old = time.time() - filephrase.JUST_WINDOW_S - 60
    for p in (home / "Downloads").iterdir():
        os.utime(p, (old, old))
    (home / "Downloads" / "second.pdf").write_bytes(b"S" * 10)
    os.utime(home / "Downloads" / "second.pdf", (old - 10, old - 10))
    m = filephrase.resolve("the PDF I just downloaded", roots=roots)
    assert m.ambiguous


def test_two_files_written_together_are_never_split_by_timestamp(tmp_path):
    import os
    (tmp_path / "Downloads").mkdir()
    now = time.time()
    for name in ("a.pdf", "b.pdf"):
        p = tmp_path / "Downloads" / name
        p.write_bytes(b"x" * 10)
        os.utime(p, (now, now))
    m = filephrase.resolve("the PDF I just downloaded",
                           roots=[str(tmp_path / "Downloads")], now=now)
    assert m.ambiguous, "same-minute arrivals are a coin flip, so ask"


def test_no_name_no_folder_no_type_is_a_question_not_a_guess(roots):
    """"email that to Heather": there is nothing here to resolve."""
    m = filephrase.resolve("that", roots=roots)
    assert m.reason == "empty" and not m.ok and not m.ambiguous


# ---- the refusals -------------------------------------------------
def test_a_symlink_out_of_the_roots_is_never_a_candidate(tmp_path, home, roots):
    secret = tmp_path / "secret.key"
    secret.write_bytes(b"PRIVATE KEY")
    (home / "Desktop" / "secret.key").symlink_to(secret)
    m = filephrase.resolve("secret key", roots=roots)
    assert not m.ok, "a symlink must not be offered at all"
    assert all("secret" not in p.name for p in m.candidates)


def test_an_explicit_path_outside_the_roots_is_allowed(tmp_path, roots):
    """He asked for "this file from this location"; a path he gives
    outright must work even outside Desktop/Downloads/Documents."""
    away = tmp_path / "projects"
    away.mkdir()
    thesis = away / "thesis.pdf"
    thesis.write_bytes(b"T" * 500)
    m = filephrase.resolve(str(thesis), roots=roots)
    assert m.ok and m.path == thesis.resolve()


def test_an_explicit_path_into_a_dot_folder_is_refused(tmp_path, roots):
    hidden = tmp_path / ".ssh"
    hidden.mkdir()
    key = hidden / "id_ed25519"
    key.write_bytes(b"PRIVATE KEY")
    m = filephrase.resolve(str(key), roots=roots)
    assert m.reason == "outside" and not m.ok


def test_an_explicit_system_path_is_refused(roots):
    m = filephrase.resolve("/etc/passwd", roots=roots)
    assert m.reason == "outside" and not m.ok


def test_a_folder_is_refused(home, roots):
    (home / "Desktop" / "coursework").mkdir()
    m = filephrase.resolve(str(home / "Desktop" / "coursework"), roots=roots)
    assert m.reason == "not-a-file"


def test_a_file_over_the_cap_is_refused_with_its_size(home, roots):
    big = home / "Desktop" / "recording.mp4"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    m = filephrase.resolve("the recording", roots=roots, max_mb=1)
    assert m.reason == "too-big" and m.size == 2 * 1024 * 1024
    line = outbox.refusal_line(m, "the recording", cap_mb=1)
    assert "megabytes" in line and "not sent" in line


def test_a_missing_explicit_path_says_so(roots):
    m = filephrase.resolve("~/Desktop/nothing-here.pdf", roots=roots)
    assert not m.ok and m.reason in ("not-found", "outside")


# ==================================================================
# 2. The recipient and the account
# ==================================================================
def test_the_contacts_map_resolves_a_name():
    addr, who = outbox.resolve_recipient(Cfg(), None, "Heather")
    assert addr == "heather@example.com" and who == "Heather"


def test_the_people_book_resolves_my_brother():
    memory = types.SimpleNamespace(
        resolve_person=lambda t: {"name": "Sam Peyrovi",
                                  "email": "sam@example.com"}
        if "brother" in t.lower() else None)
    addr, who = outbox.resolve_recipient(Cfg(), memory, "my brother")
    assert addr == "sam@example.com" and who == "Sam Peyrovi"


def test_an_unknown_name_yields_no_address_and_never_a_guess():
    memory = types.SimpleNamespace(resolve_person=lambda t: None)
    addr, who = outbox.resolve_recipient(Cfg(), memory, "Dana")
    assert addr == "" and who == "Dana"


def test_a_person_in_the_book_with_no_email_is_still_a_question():
    memory = types.SimpleNamespace(
        resolve_person=lambda t: {"name": "Dana Ruiz", "email": ""})
    addr, who = outbox.resolve_recipient(Cfg(), memory, "Dana")
    assert addr == "" and who == "Dana Ruiz"


@pytest.mark.parametrize("said,want", [
    ("heather@example.com", "heather@example.com"),
    ("heather at example dot com", "heather@example.com"),
    # A dotted LOCAL part. Reading only the domain's dots would give
    # peyrovi@tamu.edu -- a different, plausible-sounding address.
    ("h dot peyrovi at tamu dot edu", "h.peyrovi@tamu.edu"),
    ("Heather", ""),
])
def test_an_address_is_read_off_the_utterance(said, want):
    assert outbox.parse_address(said) == want


def test_one_account_needs_no_question():
    accounts = [{"label": "personal", "address": "a@b.com", "password": "p"}]
    account, why = mail_mod.choose_account(accounts)
    assert account is accounts[0] and why == ""


def test_three_accounts_and_no_hint_is_a_question():
    accounts = mail_mod.mail_accounts(Cfg())
    account, why = mail_mod.choose_account(accounts)
    assert account is None and why.startswith("which account:")
    assert "personal" in why and "work" in why and "school" in why


def test_a_spoken_hint_picks_the_account():
    accounts = mail_mod.mail_accounts(Cfg())
    account, why = mail_mod.choose_account(accounts, "school")
    assert account["address"] == "hp@tamu.edu" and why == ""


def test_an_unknown_hint_is_refused_not_approximated():
    accounts = mail_mod.mail_accounts(Cfg())
    account, why = mail_mod.choose_account(accounts, "university")
    assert account is None and why == "no account called university"


def test_a_configured_default_account_is_used():
    accounts = mail_mod.mail_accounts(Cfg())
    account, _ = mail_mod.choose_account(accounts, "", default_label="work")
    assert account["address"] == "hunter@work.com"


# ==================================================================
# 3. prepare(): the draft, or the question
# ==================================================================
def test_prepare_builds_a_draft_when_everything_resolves(roots):
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    prep = outbox.prepare(cfg, None, "the biosensors handout", "Heather")
    assert prep.ok
    d = prep.draft
    assert d.path.name == "Biosensors_Lab-Handout_v2.pdf" and d.size == 5000
    assert d.to_addr == "heather@example.com" and d.to_name == "Heather"
    assert d.account["address"] == "hp@tamu.edu"
    assert d.subject == "Biosensors Lab Handout v2"


def test_prepare_asks_which_file_before_it_asks_anything_else(roots):
    """The file is the half he named and the half most likely to be wrong,
    so an ambiguity there is the question -- not the account."""
    cfg = cfg_with_roots(roots)
    prep = outbox.prepare(cfg, None, "the lab report", "Heather")
    assert not prep.ok
    assert "lab report.pdf" in prep.ask and "lab report final.pdf" in prep.ask
    assert prep.ask.endswith("Which one?")


def test_prepare_asks_for_an_unknown_recipient(roots):
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    prep = outbox.prepare(cfg, None, "the biosensors handout", "Dana")
    assert not prep.ok and prep.ask == "I've no address for Dana, sir. What is it?"


def test_prepare_asks_which_account_with_three_configured(roots):
    cfg = cfg_with_roots(roots)
    prep = outbox.prepare(cfg, None, "the biosensors handout", "Heather")
    assert not prep.ok and "Which account" in prep.ask
    assert "personal" in prep.ask and "school" in prep.ask


def test_prepare_says_so_when_no_mailbox_is_configured(roots):
    cfg = Cfg(**{"gmail.accounts": [], "send_file.roots": roots})
    prep = outbox.prepare(cfg, None, "the biosensors handout", "Heather")
    assert not prep.ok and "gmail" in prep.ask


def test_the_cap_cannot_be_raised_by_config():
    assert outbox.max_mb(Cfg(**{"send_file.max_mb": 500})) == \
        float(outbox.MAX_ATTACHMENT_MB)
    assert outbox.max_mb(Cfg(**{"send_file.max_mb": 5})) == 5.0


# ==================================================================
# 4. The read-back
# ==================================================================
def test_the_read_back_names_file_size_address_and_account(roots):
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    draft = outbox.prepare(cfg, None, "the biosensors handout", "Heather").draft
    line = outbox.read_back(draft)
    assert "Biosensors Lab Handout v2.pdf" in line       # the FILE
    assert "5 kilobytes" in line                          # the SIZE
    assert "heather at example dot com" in line           # the ADDRESS
    assert "your school account" in line                  # the ACCOUNT
    assert line.endswith("Send it, sir?")


def test_the_address_is_spoken_not_spelled():
    assert outbox.spoken_address("h.peyrovi@tamu.edu") == \
        "h dot peyrovi at tamu dot edu"


@pytest.mark.parametrize("n,want", [
    (10, "under a kilobyte"), (3000, "3 kilobytes"),
    (2 * 1024 * 1024 + 400_000, "2.4 megabytes"),
    (40 * 1024 * 1024, "40 megabytes"),
])
def test_sizes_are_spoken_in_words(n, want):
    assert outbox.spoken_size(n) == want


# ==================================================================
# 5. The answer grammar
# ==================================================================
@pytest.mark.parametrize("said", [
    "yes", "yeah", "yep", "send it", "go ahead", "do it, jarvis",
    "yes please send it now", "confirmed", "that's right",
    # a pronoun after "to" is the person he was just read (09-04)
    "yes, send it to her", "yes send it to him", "yes send it to her please",
    "yes go ahead and send it to her", "yeah go ahead and send it",
])
def test_a_clear_yes_is_a_yes(said):
    assert parse_send_answer(said) is True


@pytest.mark.parametrize("said", [
    "no", "nope", "not that one", "wrong file", "cancel", "hold on",
    "never mind", "no thanks",
])
def test_a_clear_no_is_a_no(said):
    assert parse_send_answer(said) is False


@pytest.mark.parametrize("said", [
    # The line that earned this whole grammar: jarvis.log.1:19499, an
    # utterance that answered nothing and was read as a yes by the word-bag
    # parser, delivering a briefing nobody asked for.
    "Yeah, so you should be able to look that up.",
    "no, turn the lights off",
    "what time is it",
    "yeah I think Heather already has it",
    "okay", "sure", "alright", "mhm", "maybe",
    "yes, but send it to her work address instead",
    "yes, send it to Dana", "yes, and turn the lights off",
])
def test_nothing_ambiguous_counts_as_a_yes(said):
    assert parse_send_answer(said) is not True


def test_the_word_bag_parser_would_have_sent_it():
    """Pins WHY this grammar exists rather than reusing parse_yes_no."""
    from jarvis.commander import parse_yes_no
    stray = "Yeah, so you should be able to look that up."
    assert parse_yes_no(stray) is True
    assert parse_send_answer(stray) is None


# ==================================================================
# 6. mail.send_message with a fake transport
# ==================================================================
def _account():
    return {"label": "school", "address": "hp@tamu.edu",
            "password": "school-secret", "host": "imap.gmail.com"}


def test_send_message_builds_and_sends_the_attachment(tmp_path):
    f = tmp_path / "lab_report.pdf"
    f.write_bytes(b"%PDF-1.4 body")
    mid = mail_mod.send_message(_account(), "heather@example.com",
                                "Lab report", "Sent from Jarvis.",
                                attachment=f, smtp=FakeSMTP)
    conn = FakeSMTP.made[-1]
    assert conn.host == "smtp.gmail.com" and conn.port == 465
    assert conn.logged_in == ("hp@tamu.edu", "school-secret")
    assert conn.quit_called
    msg = conn.sent[0]
    assert msg["To"] == "heather@example.com" and msg["From"] == "hp@tamu.edu"
    assert msg["Subject"] == "Lab report" and msg["Message-ID"] == mid
    parts = [p for p in msg.iter_attachments()]
    assert len(parts) == 1
    assert parts[0].get_filename() == "lab_report.pdf"
    assert parts[0].get_payload(decode=True) == b"%PDF-1.4 body"
    assert parts[0].get_content_type() == "application/pdf"


def test_the_smtp_host_follows_the_imap_host():
    assert mail_mod.smtp_host({"host": "imap.fastmail.com"}) == "smtp.fastmail.com"
    assert mail_mod.smtp_host({"smtp_host": "mail.x.org"}) == "mail.x.org"
    assert mail_mod.smtp_host({}) == "smtp.gmail.com"


def test_a_refused_login_raises_without_the_password_in_the_message(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hi")

    class Refuses(FakeSMTP):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    with pytest.raises(mail_mod.MailSendFailed) as exc:
        mail_mod.send_message(_account(), "h@example.com", "s", "b",
                              attachment=f, smtp=Refuses)
    assert "school-secret" not in str(exc.value)
    assert str(exc.value) == "SMTPAuthenticationError"


def test_an_address_that_is_not_an_address_is_refused_before_connecting():
    with pytest.raises(mail_mod.MailSendFailed):
        mail_mod.send_message(_account(), "heather", "s", "b", smtp=FakeSMTP)
    assert not FakeSMTP.made


def test_the_suite_may_not_open_a_real_smtp_socket():
    """The conftest firewall, asserted rather than assumed. There is no
    environment escape for this one, unlike the Ollama block."""
    import socket
    import tests.conftest as ct
    before = len(ct._smtp_blocked)
    with pytest.raises(ct.SmtpFirewallRefused) as exc:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            ("smtp.gmail.com", 465))
    assert "SMTP" in str(exc.value)
    # ...and a refusal is NOT an Exception, so nothing in jarvis/ can
    # swallow it into a spoken "I couldn't send that, sir." (F24).
    assert not isinstance(exc.value, Exception)
    # This test tripped the firewall on purpose: take the entry back off
    # the list pytest_sessionfinish fails the run on.
    assert len(ct._smtp_blocked) == before + 1
    ct._smtp_blocked.pop()


def test_a_test_that_loses_its_fake_fails_loudly(cmd):
    """F24 (09-03). conftest promised "a loud failure" when a test lost its
    SMTP fake; what actually happened was ConnectionRefusedError ->
    mail.send_message's `except Exception` -> MailSendFailed -> the
    commander's ordinary "I couldn't send that, sir.", status "Send
    failed" -- and the test passed unless it happened to assert on the
    sent list. The refusal now comes OUT of handle(), through every
    `except Exception` on the way."""
    import tests.conftest as ct
    del cmd.services.smtp                               # the fake is lost
    cmd.handle("email the biosensors handout to Heather", source="typed")
    before = len(ct._smtp_blocked)
    with pytest.raises(ct.SmtpFirewallRefused):
        cmd.handle("yes", source="typed")
    assert cmd.spoken == [], "the failure must not be spoken away"
    assert len(ct._smtp_blocked) == before + 1
    ct._smtp_blocked.pop()                              # owned, see above


# ==================================================================
# 7. outbox.send(): the file must still be the file
# ==================================================================
def test_send_refuses_when_the_file_changed_after_the_read_back(roots):
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    draft = outbox.prepare(cfg, None, "the biosensors handout", "Heather").draft
    draft.path.write_bytes(b"DIFFERENT")
    with pytest.raises(outbox.DraftChanged) as exc:
        outbox.send(draft, smtp=FakeSMTP)
    assert str(exc.value) == outbox.CHANGED_LINE
    assert not FakeSMTP.made, "nothing may reach the transport"


def test_send_refuses_when_the_file_is_gone(roots):
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    draft = outbox.prepare(cfg, None, "the biosensors handout", "Heather").draft
    draft.path.unlink()
    with pytest.raises(outbox.DraftChanged) as exc:
        outbox.send(draft, smtp=FakeSMTP)
    assert str(exc.value) == outbox.GONE_LINE
    assert not FakeSMTP.made


def test_send_delivers_and_reports_who(roots):
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    draft = outbox.prepare(cfg, None, "the biosensors handout", "Heather").draft
    assert outbox.send(draft, smtp=FakeSMTP) == "Sent to Heather, sir."
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


# ==================================================================
# 8. The commander family
# ==================================================================
@pytest.fixture
def cmd(tmp_path, monkeypatch, roots):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "fb.jsonl",
                        raising=False)
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", False), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    # _bg inline: the send runs on a worker thread in production and the
    # test wants its result, not a race.
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    spoken = []
    monkeypatch.setattr(Commander, "_speak_now",
                        lambda self, text: spoken.append(text) or True)
    # Heather's book row carries a stored honorific (Hunter's 19:00 ruling,
    # 09-04: a pronoun has to match the pending person's gender, and the
    # gender comes from an EXPLICIT source only -- an honorific on the
    # person or the row, never the name). "send it to heather" still
    # resolves through the honorific-stripped key. cmd_plain below is the
    # same desk with a bare row, where her gender is unknown.
    svc = types.SimpleNamespace(
        assistant=cfg_with_roots(roots, **{
            "send_file.from": "school",
            "send_file.contacts": {"ms heather": "heather@example.com"}}),
        memory=MagicMock(), desktop=MagicMock(), workflows=MagicMock(),
        brain=MagicMock(), context=MagicMock(), tts=MagicMock(),
        timekeeper=MagicMock(), notes=MagicMock(), approvals=MagicMock(),
        claude=MagicMock(), router=MagicMock(), smtp=FakeSMTP)
    svc.desktop.parse_action = lambda part: None
    svc.workflows.get.return_value = None
    svc.context.answer_question.return_value = None
    svc.memory.resolve_person.return_value = None
    svc.memory.suggest_by_habit.return_value = None
    svc.timekeeper.ringing = None
    svc.approvals.pending.return_value = []
    svc.router.pending.return_value = None
    svc.claude.active_project = "jarvis"
    c = Commander(svc)
    c.spoken = spoken
    return c


@pytest.fixture
def cmd_plain(cmd):
    """The same desk with a bare book row: Heather's gender is UNKNOWN, so
    either pronoun confirms, exactly as before the 19:00 ruling."""
    cmd.services.assistant.data["send_file.contacts"] = {"heather": "heather@example.com"}
    return cmd


def test_the_family_is_registered_and_reachable_without_the_wake_word():
    names = [x.name for x in REGISTRY]
    assert "send file" in names
    assert "send file" in [x.name for x in ASSISTANT_TIER1]
    # After the HPCOMPUTER family, so "send the budget to HPCOMPUTER" is a
    # transfer and never an email.
    assert names.index("send file") > names.index("remote push")


def test_the_first_utterance_reads_back_and_sends_nothing(cmd):
    res = cmd.handle("email the biosensors handout to Heather", source="typed")
    assert res.handled and res.speak
    assert "Biosensors Lab Handout v2.pdf" in res.reply
    assert "heather at example dot com" in res.reply
    assert "your school account" in res.reply
    assert res.reply.endswith("Send it, sir?")
    assert not FakeSMTP.made, "the read-back must not send"
    assert cmd._pending_send is not None
    assert cmd.question_open()


def test_a_yes_then_sends_it(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("yes", source="typed")
    assert res.handled and res.ack and not res.done
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"
    assert cmd.spoken == ["Sent to Heather, sir."]
    assert cmd._pending_send is None


def test_a_voice_yes_closes_the_turn_through_the_app_door(cmd):
    """The worker's "Sent to Heather, sir." went out through services.speak
    only, and services.reply (_async_reply) is the ONE door that closes a
    done=False turn -- so after a real send the wake word stayed dead for
    the 60 s watchdog and no follow-up window opened (F20, 09-03). With the
    app's reply door present the line goes through it; without one (this
    fixture's default) the old direct door still speaks it."""
    replied = []
    cmd.services.reply = lambda text, speak=True: replied.append(text)
    cmd.handle("email the biosensors handout to Heather", source="voice")
    res = cmd.handle("yes", source="voice")
    assert res.handled and res.ack and not res.done
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"
    assert replied == ["Sent to Heather, sir."]
    assert cmd.spoken == [], "spoken twice: once per door"


def test_a_no_abandons_it(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("no", source="typed")
    assert res.reply == outbox.DROPPED_LINE
    assert not FakeSMTP.made
    assert cmd._pending_send is None


def test_a_stray_yeah_to_something_else_cannot_send_the_file(cmd):
    """The exact live line that answered a pending offer once already."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("Yeah, so you should be able to look that up.", source="typed")
    assert not FakeSMTP.made
    assert cmd._pending_send is None, "and the draft is spent, not left armed"


def test_a_new_subject_abandons_it(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("what time is it", source="typed")
    assert not FakeSMTP.made and cmd._pending_send is None
    # ...and a yes AFTER the subject changed reaches nothing.
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


def test_a_vague_answer_is_asked_again_once_and_then_dropped(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("okay", source="typed")
    assert res.reply == outbox.unsure_line(cmd._pending_send)
    assert not FakeSMTP.made
    assert cmd._pending_send is not None
    res = cmd.handle("sure", source="typed")
    assert res is None or not FakeSMTP.made
    assert cmd._pending_send is None
    assert not FakeSMTP.made


def test_an_expired_read_back_ignores_a_late_yes(cmd, monkeypatch):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd._pending_send.made_at -= outbox.DRAFT_TTL_S + 1
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made and cmd._pending_send is None


def test_an_ambiguous_file_asks_and_arms_nothing(cmd):
    res = cmd.handle("email the lab report to Heather", source="typed")
    assert "Which one?" in res.reply and cmd._pending_send is None
    assert not FakeSMTP.made


def test_an_unknown_recipient_asks_and_arms_nothing(cmd):
    res = cmd.handle("email the biosensors handout to Dana", source="typed")
    assert res.reply == "I've no address for Dana, sir. What is it?"
    assert cmd._pending_send is None and not FakeSMTP.made


def test_a_spoken_account_hint_is_honoured(cmd):
    res = cmd.handle("email the biosensors handout to Heather "
                     "from my work account", source="typed")
    assert "your work account" in res.reply
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].logged_in[0] == "hunter@work.com"


def test_send_heather_the_file_works_when_she_is_known(cmd):
    res = cmd.handle("send Heather the biosensors handout", source="typed")
    assert res is not None and res.reply.endswith("Send it, sir?")


def test_the_file_is_looked_for_with_its_original_casing(cmd, home):
    """The registry matches on lower-cased text; the NAME must not be."""
    (home / "Desktop" / "Q3-Budget.xlsx").write_bytes(b"x" * 40)
    cmd._raw_text = "email Q3-Budget.xlsx to Heather"
    res = cmd.handle("email Q3-Budget.xlsx to Heather", source="typed")
    assert "Q3 Budget.xlsx" in res.reply


# ---- the negative table -------------------------------------------
NOT_A_SEND = [
    "did you send that email to Heather",
    "i need to send the lab report to Heather",
    "i was going to email the handout to Heather",
    "send Heather a text",
    "text Heather the lab report",
    "what file is that",
    "how do i email a file",
    "file that away under biosensors",
    "reply to Heather's email",
    "forward the lab report to Heather",
    "send my location to Heather",
    "send this to discord",
    "send the next song to the speaker",
    "email Heather about the meeting",
    "send me the weather",
    "i already sent the report to Heather",
    "put the budget on HPCOMPUTER",
    "send the budget to HPCOMPUTER",
    "send the invite to Heather",
    "when should i email the handout to Heather",
    # `apolog` was anchored with \b, which never matches "apology" or
    # "apologies" -- the only stem in that alternation that was not a whole
    # word, and the only one the veto silently missed.
    "send an apology to my professor",
    "send my apologies to Heather",
]


@pytest.mark.parametrize("said", NOT_A_SEND)
def test_ordinary_sentences_never_arm_a_send(cmd, said):
    cmd.handle(said, source="typed")
    assert cmd._pending_send is None, f"{said!r} armed a file send"
    assert not FakeSMTP.made


@pytest.mark.parametrize("said", NOT_A_SEND)
def test_the_negative_table_is_the_thing_that_stops_them(said):
    """Either the grammar declines it or the negative table does — and
    which one is which is recorded here, so a future widening of the
    grammar cannot quietly remove the only guard on a phrase."""
    lowered = said.lower()
    assert (not _SEND_FILE_RX.match(lowered)) or _SEND_NOT_RX.search(lowered)


@pytest.mark.parametrize("said", [
    "email the lab report to Heather",
    "send the biosensors handout to my brother",
    "email that file on my desktop to Heather",
    "please email lab_report.pdf to Heather from my school account",
    "can you send the invoice to my advisor",
    "share the biosensors handout with Heather",
])
def test_the_real_phrasings_do_match(said):
    lowered = said.lower()
    assert _SEND_FILE_RX.match(lowered) and not _SEND_NOT_RX.search(lowered)


# ==================================================================
# 9. The model can never reach this
# ==================================================================
def test_there_is_no_tool_the_model_could_call_to_send_a_file():
    """An irreversible action must be something he asked for out loud and
    then confirmed, never something a tool loop chose. If a send_mail
    ToolSpec ever appears, this fails."""
    specs = mail_mod.make_tools(Cfg(), types.SimpleNamespace())
    assert [s.name for s in specs] == ["get_mail"]
    # And the read tool's handler cannot reach the transport: the only
    # caller of send_message in the package is jarvis/outbox.py.
    import subprocess
    root = Path(mail_mod.__file__).parent.parent
    hits = subprocess.run(["grep", "-rln", "--include=*.py",
                           "send_message", str(root)],
                          capture_output=True, text=True).stdout.split()
    assert sorted(Path(h).name for h in hits) == ["mail.py", "outbox.py"]


def test_the_shared_resolver_is_the_one_doing_the_dangerous_work():
    """filephrase must not grow its own containment or refusal logic: both
    lanes have to agree about what a path is allowed to be."""
    assert filephrase.SKIP_DIRS is filepick.SKIP_DIRS
    src = Path(filephrase.__file__).read_text()
    assert "filepick.check_file" in src
    assert "filepick.score" in src


def test_a_per_account_smtp_host_is_carried_through():
    cfg = Cfg(**{"gmail.accounts": [
        {"label": "work", "address": "h@work.com", "app_password": "s",
         "imap_host": "imap.fastmail.com", "smtp_host": "smtp.fastmail.com",
         "from_name": "Hunter Peyrovi"}]})
    account = mail_mod.mail_accounts(cfg)[0]
    assert mail_mod.smtp_host(account) == "smtp.fastmail.com"
    mail_mod.send_message(account, "h@example.com", "s", "b", smtp=FakeSMTP)
    conn = FakeSMTP.made[-1]
    assert conn.host == "smtp.fastmail.com"
    assert conn.sent[0]["From"] == "Hunter Peyrovi <h@work.com>"


def test_the_imap_rewrite_survives_the_real_config_class():
    """F22 (09-03). test_the_smtp_host_follows_the_imap_host passes on a
    bare dict; under the REAL AssistantConfig DEFAULTS carried
    gmail.smtp_host = "smtp.gmail.com", and mail_accounts copied that into
    every account that had none of its own, so the documented imap.x ->
    smtp.x rewrite was unreachable in production: a Fastmail or Exchange
    account was submitted to Gmail's server with its own credentials and
    he heard "I couldn't send that, sir." Through the config class, not a
    dict, and with a top-level smtp_host present the way load() wrote one
    into his file."""
    from jarvis.assistant_config import AssistantConfig
    cfg = AssistantConfig({"gmail": {
        "smtp_host": "smtp.gmail.com",          # what load() left in his file
        "accounts": [
            {"label": "school", "address": "hp@tamu.edu", "app_password": "s",
             "imap_host": "imap.fastmail.com"},
            {"label": "personal", "address": "h@gmail.com", "app_password": "p"},
        ]}}, None)
    school, personal = mail_mod.mail_accounts(cfg)
    assert school["host"] == "imap.fastmail.com"
    assert mail_mod.smtp_host(school) == "smtp.fastmail.com"
    assert mail_mod.smtp_host(personal) == "smtp.gmail.com"
    mail_mod.send_message(school, "h@example.com", "s", "b", smtp=FakeSMTP)
    assert FakeSMTP.made[-1].host == "smtp.fastmail.com"


def test_the_shipped_defaults_do_not_carry_a_submission_host():
    """The other half of F22: AssistantConfig.load() writes every DEFAULTS
    key into the file, so a default smtp_host is not a default, it is a
    value in his config forever. A per-account entry with no smtp_host
    and the legacy single mailbox both derive it from their IMAP host."""
    from jarvis.assistant_config import DEFAULTS, AssistantConfig
    assert "smtp_host" not in DEFAULTS["gmail"]
    legacy = AssistantConfig({"gmail": {
        "address": "h@fastmail.com", "app_password": "p",
        "imap_host": "imap.fastmail.com"}}, None)
    (single,) = mail_mod.mail_accounts(legacy)
    assert mail_mod.smtp_host(single) == "smtp.fastmail.com"


def test_arming_a_send_clears_a_read_back_that_was_already_on_the_floor(cmd):
    """Two questions cannot share one yes. The HPCOMPUTER lane parks its
    read-back in _pending_destructive; arming a send must take the floor
    from it, or the yes that sends the email leaves the push armed for
    whatever he says next."""
    ran = []
    cmd.stash_destructive(lambda: ran.append("pushed"), "Send it to HPCOMPUTER, sir?")
    cmd.handle("email the biosensors handout to Heather", source="typed")
    assert cmd._pending_destructive is None
    cmd.handle("yes", source="typed")
    assert ran == [] and FakeSMTP.made[-1].sent
    assert cmd._pending_send is None and cmd._pending_destructive is None


def test_arming_a_read_back_clears_a_send_that_was_already_on_the_floor(cmd):
    """The MIRROR of the test above, and it was missing.  stash_send cleared
    _pending_destructive and stash_destructive did not clear _pending_send,
    so with both armed "yes" sent the email and silently abandoned the
    read-back he had just heard.  What stopped it reaching handle() was an
    accident of rung ordering -- _try_send_confirm runs first and drops its
    draft on any non-answer -- not a check, and any read-back armed outside
    handle() (a proactive offer) had no such accident to rely on."""
    ran = []
    cmd.handle("email the biosensors handout to Heather", source="typed")
    assert cmd._pending_send is not None
    cmd.stash_destructive(lambda: ran.append("cancelled"),
                          "Cancel all three alarms, sir?")
    assert cmd._pending_send is None, "two questions cannot share one yes"
    cmd.handle("yes", source="typed")
    assert ran == ["cancelled"] and not FakeSMTP.made


def test_an_oversized_explicit_path_keeps_its_size_in_the_refusal(home, roots):
    big = home / "Desktop" / "dump.zip"
    big.write_bytes(b"0" * (3 * 1024 * 1024))
    m = filephrase.resolve(str(big), roots=roots, max_mb=1)
    assert m.reason == "too-big" and m.size == 3 * 1024 * 1024
    assert "3.0 megabytes" in outbox.refusal_line(m, cap_mb=1)


def test_an_empty_file_is_refused_before_the_read_back(home, roots):
    """filepick has no opinion on a 0-byte file -- it is a perfectly good
    file to copy -- but there is nothing to attach, and hearing "under a
    kilobyte ... Send it, sir?" followed by an SMTP error is the worst
    possible order to learn that in."""
    (home / "Desktop" / "blank-form.pdf").write_bytes(b"")
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    prep = outbox.prepare(cfg, None, "the blank form", "Heather")
    assert not prep.ok and prep.ask == "blank form.pdf is empty, sir; " \
        "there'd be nothing to attach."


# ==================================================================
# 10. The intent gate: what a Tier-1 name is allowed to switch off
# ==================================================================
# Registering "send file" in ASSISTANT_TIER1 is what makes the feature
# reachable by voice at all (the hotword eats the wake word). It also turns
# the intent classifier OFF for every sentence the matcher accepts -- so the
# matcher standing there IS the gate for the one irreversible family in the
# app, and the raw grammar was much too generous to be it.
GATE_MUST_NOT_CLAIM = [
    # vetoed by the negative table, which used to run only AFTER the gate
    # had already been switched off -- so these were handed to the model
    # out loud instead of being dropped as background chat
    "send my regards to Heather",
    "send a text to Heather",
    "send the money to Ali",
    "send an apology to my professor",
    # named nobody he can write to and no file that exists: 28 of 33
    # everyday sentences like these were claimed, and answered with a
    # file-shaped refusal ("I can't find a file by that name, sir.")
    "send the kids to bed",
    "send the package to my dad",
    "send flowers to my mom",
    "send the deposit to the landlord",
    "send a card to my grandmother",
    "share the road with cyclists",
    "share my screen with the class",
    "send that to the printer",
    "send it to the shop",
    "share that with the group",
]


@pytest.mark.parametrize("said", GATE_MUST_NOT_CLAIM)
def test_an_everyday_sentence_never_reaches_the_send_lane_by_voice(cmd, said):
    """Bare voice, no wake word -- the live path. Nothing is armed, and the
    words are NOT claimed: they go to the classifier like any other
    overheard sentence."""
    res = cmd.handle(said, source="voice")
    assert cmd._pending_send is None, f"{said!r} armed a file send"
    assert not FakeSMTP.made
    assert res.status in ("Ignored (background chat)", "Was that for me?"), \
        f"{said!r} was claimed: {res.status!r} / {res.reply!r}"


@pytest.mark.parametrize("said", GATE_MUST_NOT_CLAIM)
def test_and_the_tier_1_probe_agrees(cmd, said):
    """The gate consults _match_assistant, so the two must not disagree."""
    assert cmd._match_assistant(said.lower()) != "send file"


def test_a_real_request_still_works_by_voice_with_no_wake_word(cmd):
    """The reason the name is in ASSISTANT_TIER1 at all."""
    res = cmd.handle("email the biosensors handout to Heather", source="voice")
    assert res.reply.endswith("Send it, sir?")
    assert cmd._pending_send is not None


def test_a_real_file_makes_it_a_request_even_when_the_person_is_unknown(cmd):
    """"Email the biosensors handout to Dana" names a file that is really on
    his disk, so he was plainly talking to me: it asks for Dana's address
    rather than being dropped with the household sentences above."""
    res = cmd.handle("email the biosensors handout to Dana", source="voice")
    assert res.reply == "I've no address for Dana, sir. What is it?"
    assert cmd._pending_send is None


def test_a_backchannel_yeah_cannot_send_what_was_never_armed(cmd):
    """The whole exposure in one test: unaddressed speech of the right
    shape, then a backchannel "yeah"."""
    cmd.handle("send the deposit to the landlord", source="voice")
    cmd.handle("yeah", source="voice")
    assert not FakeSMTP.made


# ==================================================================
# 11. A yes has to come from the room the question was asked in
# ==================================================================
@pytest.mark.parametrize("src", ["discord", "phone", "socket", "cli"])
def test_a_yes_from_a_channel_that_never_heard_the_question_sends_nothing(
        cmd, src):
    """app.py publishes UserUtterance(source="discord") for a Discord
    message and `phone` carries a bearer token in SECRET_KEYS, so these are
    live channels, not hypotheticals. A read-back spoken at the desk cannot
    be answered from one of them."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("yes", source=src)
    assert not FakeSMTP.made, f"a {src} yes sent the attachment"
    assert res is None or res.status != "Sending"
    # ...and it is not a cancellation either: the question is still his.
    assert cmd._pending_send is not None
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_the_desk_is_one_room(cmd):
    """Voice and typed are the same window; he may answer either way."""
    cmd.handle("email the biosensors handout to Heather", source="voice")
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent


# ==================================================================
# 12. A question somebody else answered spends the draft
# ==================================================================
def test_an_offer_answered_in_between_takes_the_draft_with_it(cmd):
    """The alarm offer, the study offer, the briefing offer and a ringing
    timer all arrive OUT OF BAND and all sit above the send rung. Measured:
    the yes that answered the offer left the draft armed for the rest of
    its 90 s, and the next yes-shaped utterance -- aimed at anything --
    sent the file."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.services.alarm_offer = {"made_at": time.time(), "when": 7 * 3600,
                                "label": "wake"}
    cmd.handle("yes", source="typed")
    assert cmd._pending_send is None, "the offer's yes left the draft armed"
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


def test_a_ringing_timer_takes_the_draft_with_it(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.services.timekeeper.ringing = types.SimpleNamespace(label="tea")
    cmd.handle("stop", source="typed")
    assert cmd._pending_send is None
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


# ==================================================================
# 13. "Which one, sir?" is a question, and questions get answered
# ==================================================================
@pytest.mark.parametrize("answer", [
    "the final one", "the second one", "lab report final", "final",
    "lab_report_final.pdf", "the last one",
])
def test_the_ambiguous_branch_hears_its_own_answer(cmd, answer):
    """The designed-for common case -- lab_report.pdf against
    lab_report_final.pdf -- used to be a silent dead end: nothing was
    parked, question_open() said no, the follow-up microphone got the short
    window and every natural answer was called background chat."""
    res = cmd.handle("email the lab report to Heather", source="typed")
    assert "Which one?" in res.reply
    assert cmd.question_open(), "the mic must stay open for the answer"
    res = cmd.handle(answer, source="typed")
    assert res is not None and res.reply.endswith("Send it, sir?")
    assert "lab report final.pdf" in res.reply
    assert not FakeSMTP.made, "choosing a file confirms nothing"
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent


def test_the_first_one_means_the_first_one(cmd):
    cmd.handle("email the lab report to Heather", source="typed")
    res = cmd.handle("the first one", source="typed")
    assert "lab report.pdf" in res.reply and "final" not in res.reply


def test_neither_of_them_drops_the_question_out_loud(cmd):
    cmd.handle("email the lab report to Heather", source="typed")
    res = cmd.handle("neither", source="typed")
    assert res.reply == "Very good, sir." and not cmd.question_open()


def test_an_answer_that_still_fits_both_is_asked_again_not_guessed(cmd):
    """Repeating the ambiguous phrase is not a choice.  The margin a name
    has to beat is filephrase.TIE_SCORE -- the very band that called these
    two ambiguous -- so "the lab report" cannot resolve to the higher
    scorer, and one more question is asked rather than a guess made or a
    silence returned.  The second near miss spends it."""
    cmd.handle("email the lab report to Heather", source="typed")
    res = cmd.handle("the lab report", source="typed")
    assert res is not None and "Which of them?" in res.reply
    assert cmd.question_open() and not FakeSMTP.made
    # ...and now the ordinal lands
    res = cmd.handle("the second one", source="typed")
    assert "lab report final.pdf" in res.reply


def test_the_second_near_miss_spends_the_question(cmd):
    cmd.handle("email the lab report to Heather", source="typed")
    cmd.handle("the lab report", source="typed")
    cmd.handle("the lab report", source="typed")
    assert cmd._pending_filepick is None and not FakeSMTP.made


def test_an_answer_that_names_none_of_them_is_not_a_pick(cmd):
    """A wrong pick would send the wrong file, so a name that does not beat
    its rivals clearly is not an answer at all."""
    cmd.handle("email the lab report to Heather", source="typed")
    cmd.handle("what time is it", source="typed")
    assert cmd._pending_filepick is None and not FakeSMTP.made


def test_a_bare_yes_is_never_a_choice_between_two_files(cmd):
    cmd.handle("email the lab report to Heather", source="typed")
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made and cmd._pending_send is None


# ==================================================================
# 14. The yeses that used to be dropped in silence
# ==================================================================
@pytest.mark.parametrize("said", [
    "system, yes.",              # real, jarvis.log.1:5313
    "yes it is", "I think so yes",
])
def test_a_yes_this_grammar_does_not_take_is_asked_again_not_dropped(cmd, said):
    """parse_yes_no reads all of these as yes and parse_send_answer does
    not. Sending on them is how a file reaches the wrong person; dropping
    them without a word is how he learns the feature does not work. They
    get the re-ask the vague fillers already got.

    (09-04, the third review: "um, yes", "uh yeah", "okay yes", "mhm
    yeah", "well, yes" and "yes that's the one" used to sit in this list.
    A filler in front of a plain yes is not doubt, and eighteen of his
    plain yeses were being re-asked; they are yeses now -- section 21.)"""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None
    assert res.reply == outbox.unsure_line(cmd._pending_send)
    assert not FakeSMTP.made and cmd._pending_send is not None
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent


def test_a_full_stop_no_longer_breaks_a_clean_yes(cmd):
    """"Yes. Thank you." is one of his own logged answers
    (jarvis.log.1:17109); the tail separator allowed only a comma or a
    space, so the full stop ended the sentence and a clean yes fell
    through to the silent-drop branch."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("Yes. Thank you.", source="typed")
    assert res.ack and FakeSMTP.made[-1].sent


def test_the_ten_word_stray_is_still_dropped_in_silence(cmd):
    """The widening must NOT reach the sentence the whole grammar exists
    for. Six words is parse_yes_no's own overheard-speech line."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("Yeah, so you should be able to look that up.",
                     source="typed")
    assert res is None or not str(res.reply or "").startswith("I'd rather be certain")
    assert not FakeSMTP.made and cmd._pending_send is None


# ==================================================================
# 15. Two sends in one breath
# ==================================================================
def test_two_sends_in_one_breath_are_not_one_long_recipient(cmd):
    """_compound_hijack bailed whenever a matcher accepted BOTH clauses, so
    the whole-utterance match stood: _SEND_FILE_RX's `to`-split is
    non-greedy but $-anchored, and who_a swallowed the second clause.
    Jarvis asked for the address of "Heather and email the biosensors
    handout to Heather"."""
    res = cmd.handle("jarvis email the notes to Heather and email the "
                     "biosensors handout to Heather", source="typed")
    assert "and email" not in (res.reply or "")
    assert not FakeSMTP.made
    # One read-back is on the table, and it names a real file.
    assert cmd._pending_send is not None
    assert "One at a time" in (res.status or "")


def test_a_second_send_never_quietly_replaces_the_first(cmd):
    """Two read-backs, one answerable slot: the second is refused out loud
    rather than overwriting a question he has already heard."""
    from jarvis.commander import _SEND_FILE_RX, _h_send_file
    cmd.handle("email the biosensors handout to Heather", source="typed")
    first = cmd._pending_send
    # The handler called directly, which is what a compound clause does:
    # every path through handle() spends the draft in _try_send_confirm
    # before a second send can arm.
    text = "email the invoice to heather"
    cmd._raw_text = text
    res = _h_send_file(cmd, text, _SEND_FILE_RX.match(text))
    assert res.status == "One at a time"
    assert cmd._pending_send is first
    assert not FakeSMTP.made


# ==================================================================
# 16. Which identity, and how sure we have to be
# ==================================================================
@pytest.mark.parametrize("hint", ["s", "w", "p", "hp", "hunter", "worked",
                                  "sc", "sch00l"])
def test_a_hint_that_is_not_a_name_picks_no_identity(hint):
    """Sending as the wrong one of his three identities is one of the two
    irreversible halves of this feature, and the docstring has always said
    an unrecognised hint must produce a question rather than a near miss.
    Measured before the fix: "s" -> school, "w" -> work, "p" -> personal,
    "worked" -> work, and the local-part leg made "hp" -- the first half of
    HPCOMPUTER -- his school account."""
    accounts = mail_mod.mail_accounts(Cfg())
    assert mail_mod.account_by_label(accounts, hint) is None


@pytest.mark.parametrize("hint,label", [
    ("school", "school"), ("work", "work"), ("personal", "personal"),
    ("sch", "school"), ("hp@tamu.edu", "school"),
])
def test_a_hint_that_is_a_name_still_resolves(hint, label):
    accounts = mail_mod.mail_accounts(Cfg())
    picked = mail_mod.account_by_label(accounts, hint)
    assert picked is not None and mail_mod.account_label(picked) == label


def test_an_unrecognised_hint_asks(cmd):
    res = cmd.handle("email the biosensors handout to Heather from my s "
                     "account", source="typed")
    assert cmd._pending_send is None and not FakeSMTP.made
    assert "no s account" in (res.reply or "").lower()


# ==================================================================
# 16. "Which account?" and "What is it?" are questions too (F21, 09-03)
# ==================================================================
# His real config has three accounts and send_file.from ships BLANK, so
# on the shipped default EVERY send that does not say "from my X account"
# stopped at "Which account should I send from, sir — personal, work,
# school?" -- spoken with no slot behind it. question_open() was False,
# the app gave the answer the 4 s window and the intent gate, and
# "school" came back as "Was that for me?". Same for "I've no address for
# Dana, sir. What is it?": the address he then said went nowhere.
@pytest.fixture
def cmd_no_default(cmd):
    """The shipped default: three accounts and no send_file.from."""
    cmd.services.assistant.data.pop("send_file.from", None)
    return cmd


def test_which_account_is_a_question_he_can_answer(cmd_no_default):
    cmd = cmd_no_default
    res = cmd.handle("email the biosensors handout to Heather", source="voice")
    assert res.reply.startswith("Which account should I send from, sir")
    assert cmd._pending_send is None and cmd._pending_sendask is not None
    assert cmd.question_open(), "the answer gets the question window"
    res = cmd.handle("school", source="voice")
    assert res is not None and res.reply.endswith("Send it, sir?"), res
    assert "your school account" in res.reply
    assert "heather at example dot com" in res.reply
    assert cmd._pending_sendask is None and cmd._pending_send is not None
    assert not FakeSMTP.made, "answering a question never sends"
    cmd.handle("yes", source="voice")
    assert FakeSMTP.made[-1].logged_in[0] == "hp@tamu.edu"
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


@pytest.mark.parametrize("said", [
    "work", "the work one", "from my work account", "use work, please",
    "Work.", "my work account", "send it from work", "the work account",
])
def test_the_account_answer_takes_his_phrasings(cmd_no_default, said):
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    res = cmd.handle(said, source="voice")
    assert res is not None and "your work account" in res.reply, (said, res)
    assert cmd._pending_send is not None and not FakeSMTP.made


def test_a_label_he_does_not_have_is_asked_once_more_then_let_go(cmd_no_default):
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    res = cmd.handle("yahoo", source="voice")
    assert res.reply == outbox.ACCOUNT_REASK_LINE.format(
        hint="yahoo", names="personal, work, school")
    assert cmd.question_open(), "asked again: still a question"
    res = cmd.handle("hotmail", source="voice")
    assert res.reply == outbox.ASK_DROPPED_LINE
    assert cmd._pending_sendask is None and cmd._pending_send is None
    assert not FakeSMTP.made


def test_a_bare_yes_is_not_an_account(cmd_no_default):
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    res = cmd.handle("yes", source="voice")
    assert res.reply == outbox.ACCOUNT_WHICH_LINE.format(
        names="personal, work, school")
    res = cmd.handle("personal", source="voice")
    assert "your personal account" in res.reply and not FakeSMTP.made


def test_a_no_to_the_account_question_lets_it_go_out_loud(cmd_no_default):
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    res = cmd.handle("never mind", source="voice")
    assert res.reply == outbox.ASK_SPENT_LINE
    assert cmd._pending_sendask is None and not cmd.question_open()


def test_a_new_subject_drops_the_account_question(cmd_no_default):
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    cmd.handle("what time is it", source="voice")
    assert cmd._pending_sendask is None and cmd._pending_send is None
    cmd.handle("school", source="voice")
    assert cmd._pending_send is None, "the question is gone; 'school' is not its answer"


def test_the_account_question_is_not_answered_from_another_room(cmd_no_default):
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    cmd.handle("school", source="discord")
    assert cmd._pending_sendask is not None, "parked, not spent"
    assert cmd._pending_send is None
    res = cmd.handle("school", source="voice")
    assert "your school account" in res.reply


def test_an_expired_account_question_ignores_a_late_answer(cmd_no_default):
    from jarvis.commander import SENDASK_TTL_S
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    cmd._pending_sendask.made_at -= SENDASK_TTL_S + 1
    assert not cmd.question_open()
    cmd.handle("school", source="voice")
    assert cmd._pending_send is None and cmd._pending_sendask is None


def test_an_offer_answered_in_between_takes_the_question_with_it(cmd_no_default):
    """Same rule the draft already keeps (_drop_stranded_questions)."""
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    cmd.services.alarm_offer = {"made_at": time.time(), "when": 7 * 3600,
                                "label": "wake"}
    cmd.handle("yes", source="voice")
    assert cmd._pending_sendask is None, "the offer's yes left the question armed"


def test_whats_the_address_is_a_question_he_can_answer(cmd):
    res = cmd.handle("email the biosensors handout to Dana", source="voice")
    assert res.reply == "I've no address for Dana, sir. What is it?"
    assert cmd.question_open() and cmd._pending_sendask.kind == "recipient"
    res = cmd.handle("dana at example dot com", source="voice")
    assert res.reply.endswith("Send it, sir?"), res
    assert "to Dana, at dana at example dot com" in res.reply
    assert not FakeSMTP.made
    cmd.handle("yes", source="voice")
    assert FakeSMTP.made[-1].sent[0]["To"] == "dana@example.com"


@pytest.mark.parametrize("said", [
    "it's dana at example dot com", "send it to dana@example.com",
    "her address is dana at example dot com", "dana@example.com, please",
])
def test_the_address_answer_takes_his_phrasings(cmd, said):
    cmd.handle("email the biosensors handout to Dana", source="voice")
    res = cmd.handle(said, source="voice")
    assert res is not None and "dana at example dot com" in res.reply, (said, res)
    assert cmd._pending_send is not None and not FakeSMTP.made


def test_a_name_the_book_knows_answers_the_address_question(cmd):
    cmd.handle("email the biosensors handout to Dana", source="voice")
    res = cmd.handle("send it to Heather instead", source="voice")
    assert "to Heather, at heather at example dot com" in res.reply
    assert not FakeSMTP.made


def test_a_non_address_is_asked_once_more_then_let_go(cmd):
    cmd.handle("email the biosensors handout to Dana", source="voice")
    res = cmd.handle("dana at example com", source="voice")   # no "dot"
    assert res.reply == outbox.ADDRESS_REASK_LINE and cmd.question_open()
    res = cmd.handle("dana example", source="voice")
    assert res.reply == outbox.ASK_DROPPED_LINE
    assert cmd._pending_sendask is None and not FakeSMTP.made


def test_who_should_i_send_it_to_is_answerable_as_well(cmd):
    """WHO_LINE, the empty-recipient shape of the same question."""
    from jarvis.commander import SendAsk
    ask = SendAsk(kind="recipient", said_file="the biosensors handout",
                  who="", hint="school")
    cmd._turn_source = "voice"
    cmd.stash_sendask(ask)
    assert cmd.question_open()
    res = cmd.handle("Heather", source="voice")
    assert "to Heather, at heather at example dot com" in res.reply


def test_the_question_slots_never_share_a_floor(cmd_no_default):
    """Arming the account question clears a read-back and vice versa, the
    invariant every other slot pair keeps."""
    from jarvis.commander import SendAsk
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather "
               "from my school account", source="voice")
    assert cmd._pending_send is not None
    cmd.stash_sendask(SendAsk(kind="account", said_file="x", who="Heather", hint=""))
    assert cmd._pending_send is None and cmd._pending_sendask is not None
    cmd.handle("email the biosensors handout to Heather "
               "from my school account", source="voice")
    assert cmd._pending_sendask is None and cmd._pending_send is not None
    cmd.stash_sendask(SendAsk(kind="account", said_file="x", who="Heather", hint=""))
    cmd.stash_filepick([Path("/tmp/a"), Path("/tmp/b")], lambda p: None)
    assert cmd._pending_sendask is None


# ==================================================================
# 17. A yes that carries a correction (F23, 09-03)
# ==================================================================
def test_a_yes_with_a_new_recipient_never_sends_to_the_old_one(cmd):
    """Read-back to Heather; "yes, send it to Dana". parse_send_answer said
    None, the vague leg re-asked the ORIGINAL question, and the next bare
    "yes" sent the file to Heather -- the correction discarded without a
    word. Now the draft is spent and the SAME file is read back again to
    Dana; with no address for her, that is the address question."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("yes, send it to Dana", source="typed")
    assert res.reply == "I've no address for Dana, sir. What is it?"
    assert cmd._pending_send is None and not FakeSMTP.made
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made, "a yes after the correction must not send to Heather"
    res = cmd.handle("dana at example dot com", source="typed")
    assert "to Dana, at dana at example dot com" in res.reply
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "dana@example.com"


@pytest.mark.parametrize("said", [
    "yes, send it to Dana", "yes but to Dana", "yeah, to Dana instead",
    "yes, but send it to Dana instead", "yes send it to dana@example.com",
    "yes, actually send it to Dana please",
])
def test_a_corrected_recipient_the_book_knows_is_read_back_again(cmd, said):
    cmd.services.assistant.data["send_file.contacts"]["dana"] = "dana@example.com"
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.reply.endswith("Send it, sir?"), (said, res)
    assert "dana at example dot com" in res.reply and "heather" not in res.reply
    assert not FakeSMTP.made
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "dana@example.com"


def test_a_yes_with_an_account_correction_is_read_back_from_that_account(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("yes, but from my work account", source="typed")
    assert res.reply.endswith("Send it, sir?")
    assert "your work account" in res.reply
    assert "to Heather, at heather at example dot com" in res.reply
    assert not FakeSMTP.made
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].logged_in[0] == "hunter@work.com"


def test_the_seven_word_correction_is_no_longer_a_silent_drop(cmd):
    """The parametrised silent-drop case the old test relied on: "yes, but
    send it to her work address instead" fell to the drop branch and he
    heard nothing at all. It is a correction; it gets the address question."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("yes, but send it to her work address instead",
                     source="typed")
    assert res is not None and res.reply.endswith("What is it?"), res
    assert cmd.question_open() and not FakeSMTP.made
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


def test_the_re_ask_names_the_file_and_the_recipient_again(cmd):
    """The second half of F23: the generic re-ask asked for a yes without
    saying what the yes was to."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("okay", source="typed")
    assert "Biosensors Lab Handout v2.pdf" in res.reply
    assert "Heather, at heather at example dot com" in res.reply
    assert res.reply.startswith("I'd rather be certain")
    assert not FakeSMTP.made and cmd._pending_send is not None


def test_the_ten_word_stray_is_not_a_correction_either(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("Yeah, so you should be able to look that up.", source="typed")
    assert cmd._pending_send is None and cmd._pending_sendask is None
    assert not FakeSMTP.made


@pytest.mark.parametrize("said,want", [
    ("yes, send it to Dana", ("Dana", "")),
    ("yes but to her work address instead", ("work address", "")),
    ("yes, from my work account", ("", "work")),
    ("yes send it to dana@example.com from my school account",
     ("dana@example.com", "school")),
    ("yes", None), ("yes please", None), ("yes that's the one", None),
    ("Yeah, so you should be able to look that up.", None),
    ("yes, thank you", None),
])
def test_what_counts_as_a_correction(said, want):
    from jarvis.commander import _send_correction
    assert _send_correction(said) == want, said


# ==================================================================
# 18. A pronoun after "to" is a yes, not a new recipient (09-04)
# ==================================================================
@pytest.mark.parametrize("said", [
    # ("yes send it to him" moved to GENDER_REASKS, section 24: Heather's
    # row carries an honorific now, and "him" is not her -- 19:00 ruling)
    "yes, send it to her",
    "yes go ahead and send it to her", "yes send it to her please",
])
def test_a_yes_that_names_the_recipient_by_pronoun_sends(cmd, said):
    """Read-back to Heather; "yes, send it to her". The F23 correction
    grammar took the bare pronoun as a NEW recipient called "her" and
    asked "I've no address for her, sir. What is it?" -- on the send path,
    where the read-back is the whole safety story. A pronoun after "to"
    refers to the person he has just heard named: it is the yes."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled, (said, res)
    assert "no address" not in res.reply.lower(), (said, res.reply)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said
    assert cmd._pending_send is None and cmd._pending_sendask is None


@pytest.mark.parametrize("said", [
    "yes, send it to her", "yes send it to him", "yes, to them",
    "yes go ahead and send it to her", "yes send it to her please",
    "yes send it to me", "yes, send that to it",
])
def test_a_pronoun_is_never_a_corrected_recipient(said):
    from jarvis.commander import _send_correction
    assert _send_correction(said) is None, said


def test_a_possessive_pronoun_still_corrects_the_address(cmd):
    """The neighbour that must keep working: "her work address" names an
    address of the person, not the person."""
    from jarvis.commander import _send_correction
    assert _send_correction("yes but to her work address instead") == ("work address", "")


# ==================================================================
# 19. The one re-ask belongs to ONE question (09-04)
# ==================================================================
def test_a_re_asked_address_does_not_spend_the_account_question(cmd_no_default):
    """Shipped default: three accounts, no send_file.from, and Dana is not
    in the book. The address question is re-asked once; the corrected
    answer then reached "Which account?" carrying reasked=True, and the
    account question -- never yet asked -- was let go as "asked twice".
    The read-back's own re-ask must be untouched by either."""
    cmd = cmd_no_default
    res = cmd.handle("email the biosensors handout to Dana", source="voice")
    assert res.reply == "I've no address for Dana, sir. What is it?"
    res = cmd.handle("dana at example com", source="voice")       # no "dot"
    assert res.reply == outbox.ADDRESS_REASK_LINE
    res = cmd.handle("dana at example dot com", source="voice")
    assert res.reply.startswith("Which account should I send from, sir"), res
    assert cmd._pending_sendask is not None and cmd.question_open()
    res = cmd.handle("school", source="voice")
    assert res.reply.endswith("Send it, sir?"), res
    # (b) the name he said survives the account question in between
    assert "to Dana, at dana at example dot com" in res.reply
    assert "your school account" in res.reply
    assert not FakeSMTP.made
    # the read-back keeps its own single re-ask
    res = cmd.handle("okay", source="voice")
    assert res.reply.startswith("I'd rather be certain") and cmd._pending_send is not None
    cmd.handle("yes", source="voice")
    assert FakeSMTP.made[-1].sent[0]["To"] == "dana@example.com"
    assert FakeSMTP.made[-1].logged_in[0] == "hp@tamu.edu"


def test_a_re_asked_account_does_not_spend_the_read_back(cmd_no_default):
    cmd = cmd_no_default
    cmd.handle("email the biosensors handout to Heather", source="voice")
    res = cmd.handle("yes", source="voice")                    # not an account
    assert res.reply.startswith("Which of them, sir") and cmd._pending_sendask.reasked
    res = cmd.handle("school", source="voice")
    assert res.reply.endswith("Send it, sir?"), res
    res = cmd.handle("okay", source="voice")
    assert res.reply.startswith("I'd rather be certain") and cmd._pending_send is not None
    cmd.handle("yes", source="voice")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


# ==================================================================
# 20. The second review (09-04): a pronoun or demonstrative after "to",
#     with or without a noun on it, is the draft said back -- a yes.
#     Only a NAME or an ADDRESS after "to" corrects.
# ==================================================================
SEND_CORPUS = [
    # the literal echo of the read-back, on its own
    # ("send it to him please", "yes send it to him please", "yeah, send it
    # to him" and "yes, send it to his inbox" moved to GENDER_REASKS in
    # section 24 -- Hunter's 19:00 ruling: "him" is not Heather)
    "send it to her", "please send it to her",
    "jarvis, send it to her", "send it to her, jarvis", "send that to her now",
    "send it over to her", "Send it to her.",
    # a yes that names the recipient by pronoun
    "yes, send it to her",
    "yes please send it to her", "yes, send it to them", "yes, to her",
    "yes go ahead and send it to her", "Yes. Send it to her.",
    "correct, send it to her", "yes, send it to her, thanks",
    # the possessive plus a noun that is still the same person
    "yes send it to her address", "yes, send it to her email",
    "yes, send it to her inbox", "yes send it to her email address",
    "yes send it to their address",
    "yes to her address please",
    # the demonstrative forms the spec listed
    "yes, to that address", "yes to this address", "yes, send it to that address",
    "yes, send it to the same address", "yes, the same address", "yes, that one",
    "yes, that person", "yes to that person", "yes send it to the same one",
    "yes, to the same person", "yep send it to that address",
    # the account he was just read is not a new account either
    "yes, from the same account", "yes, send it from that account",
    # a filler in front of the read-back's own words (09-04, third review)
    "okay send it to her",
]


@pytest.mark.parametrize("said", SEND_CORPUS)
def test_a_confirmation_shaped_answer_sends_to_the_pending_address(cmd, said):
    """Read-back to Heather. "yes send it to her address" stripped the
    possessive BEFORE the pronoun check, so "address" became a corrected
    recipient and he heard "I've no address for address, sir". "send it
    to her" -- the read-back's own words -- was in the yes TAIL only and
    was not a yes on its own. Every sentence here is the draft said back,
    and the draft goes to the address he was read."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled, (said, res)
    assert "no address" not in res.reply.lower(), (said, res.reply)
    assert "rather be certain" not in res.reply, (said, res.reply)
    assert FakeSMTP.made, (said, res.reply)
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said
    assert cmd._pending_send is None and cmd._pending_sendask is None


@pytest.mark.parametrize("said", SEND_CORPUS)
def test_a_confirmation_shaped_answer_is_never_a_correction(said):
    from jarvis.commander import _send_correction
    assert _send_correction(said) is None, said


NOT_SEND_CORPUS = [
    # a NAME or an ADDRESS after "to" is a correction, never a yes
    "yes, send it to Dana", "yes send it to dana@example.com",
    "yes, but to her work address instead", "send it to Dana",
    # an account by name is a correction too
    "yes, but from my work account",
    # a second command riding on the yes is not a yes
    "yes, and turn the lights off", "send it to her and turn the lights off",
    # no, in every shape, including the contradictory one
    "no", "no, don't send it to her", "no, send it to her", "not to her",
    # the fillers ALONE stay vague: one re-ask, nothing sent ("okay send it
    # to her" moved to SEND_CORPUS, 09-04: the okay is a prefix on the
    # read-back's own words)
    "sure", "okay",
    # not an answer at all
    "Yeah, so you should be able to look that up.", "what's the weather",
    "send it to her tomorrow",
]


SELF_REDIRECTS = [
    "send it to me", "send it to us", "send it to you", "yes, send it to me",
    "send it to me please", "yes to me", "send that to me", "jarvis send it to me",
    "send it over to me", "send it to me and her",
]


@pytest.mark.parametrize("said", SELF_REDIRECTS)
def test_a_redirect_to_himself_never_sends_and_is_answered(cmd, said):
    """09-04, the independent post-merge review: every one of these SENT the
    file to the person he had been read. me/us/you sat in the pronoun set
    beside her/him, so "send it to me" was the read-back echoed. It is a
    redirect: answered with whom the draft is to, draft kept, nothing sent."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    FakeSMTP.made.clear()
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled, (said, res)
    assert "not to you" in res.reply and "Heather" in res.reply, (said, res.reply)
    assert not any(getattr(c, "sent", None) for c in FakeSMTP.made), said
    assert cmd._pending_send is not None, "the draft must stay armed"
    # ...and a real yes afterwards still sends to the person he was read.
    res2 = cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", (said, res2)


@pytest.mark.parametrize("said", NOT_SEND_CORPUS)
def test_a_name_a_command_or_a_no_never_sends_to_the_pending_address(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle(said, source="typed")
    assert not FakeSMTP.made, said


@pytest.mark.parametrize("said", [
    "yes, send it to Dana", "yes send it to dana@example.com",
    "yes, but to her work address instead", "yes, but from my work account",
])
def test_a_name_an_address_or_an_account_still_corrects(said):
    """What must keep holding: a correction is still a correction."""
    from jarvis.commander import _send_correction
    fix = _send_correction(said)
    assert fix is not None and fix != ("", ""), (said, fix)


@pytest.mark.parametrize("said", [
    "yes, send it to Dana", "yes, and turn the lights off", "okay", "sure",
    "send it to her tomorrow",
])
def test_the_guards_are_not_a_yes(said):
    assert parse_send_answer(said) is not True, said


def test_no_is_tested_before_yes_on_the_contradictory_sentence():
    assert parse_send_answer("no, send it to her") is False


def test_the_fillers_get_one_re_ask_with_a_pronoun_on_them(cmd):
    """A filler ALONE is still the one re-ask; the read-back's own words
    after it then send. (Until 09-04 this test said "okay send it to her"
    was the re-ask too; the okay is a prefix on a plain yes -- section 21.)"""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("okay", source="typed")
    assert res.reply.startswith("I'd rather be certain") and cmd._pending_send is not None
    assert not FakeSMTP.made
    cmd.handle("send it to her", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_her_address_is_not_a_recipient_called_address(cmd):
    """The review's own sentence, end to end."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("yes send it to her address", source="typed")
    assert "address for address" not in res.reply, res.reply
    assert res.ack and cmd.spoken == ["Sent to Heather, sir."], (res.reply, cmd.spoken)
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


# ==================================================================
# 21. The third review (09-04): fillers, doubled yeses, the whole-phrase
#     yeses -- and while a draft is pending, NOTHING returns silence.
# ==================================================================
# Measured on this harness before the fix: the first eighteen were RE-ASKED
# ("I'd rather be certain...") and the last seven got no reply at all --
# handled=True, reply=None, draft gone, the sentence handed to the chat
# model. Every one is a plain yes to "...to Heather. Send it, sir?".
THIRD_REVIEW_SENDS = [
    # re-asked: fillers and doubled yeses in front of a plain yes
    "yes that's right, send it", "that's correct, send it", "yes yes yes",
    "um yes send it to her", "uh, yes", "er, yes, send it", "hmm yes",
    "yes yes", "yes yes send it", "yeah yeah send it", "yes, yes, to her",
    "yes, that's the one", "that one, yes", "right, to the same address",
    "yep, send it over", "yes, go for it", "yes, ship it", "yeah sure send it",
    # silent: no yes-word at all, and a whole phrase the grammar lacked
    "the same one", "go on", "go on then", "send it over", "send it along",
    "send it off", "that address please",
    # the ones section 14 used to pin as re-asks
    "um, yes", "uh yeah", "okay yes", "mhm yeah", "well, yes",
]


@pytest.mark.parametrize("said", THIRD_REVIEW_SENDS)
def test_the_third_review_yeses_send_to_the_pending_address(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled and res.ack, (said, res)
    assert "rather be certain" not in str(res.reply), (said, res.reply)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent, (said, res.reply)
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said
    assert cmd._pending_send is None and cmd._send_aside is None


@pytest.mark.parametrize("said", THIRD_REVIEW_SENDS)
def test_the_third_review_yeses_are_a_yes_to_the_grammar(said):
    assert parse_send_answer(said) is True, said


@pytest.mark.parametrize("said,want", [
    ("um yes send it to her", "yes send it to her"),
    ("yes yes yes", "yes"),
    ("yes, yes, to her", "yes, to her"),
    ("yeah sure send it", "yeah send it"),
    ("jarvis, um, yes", "jarvis, yes"),
    ("right, to the same address", "to the same address"),
    ("well, no", "no"),
    # nothing left after the fillers come off: handed back whole
    ("okay", "okay"), ("sure", "sure"), ("okay, okay.", "okay, okay."),
    # a word that merely contains a filler is untouched
    ("her address", "her address"),
])
def test_what_send_clean_takes_off(said, want):
    from jarvis.commander import _send_clean
    assert _send_clean(said) == want


# ---- the guards the review will attack ---------------------------------
@pytest.mark.parametrize("said", [
    # NO still wins, with or without a filler or a new head after it
    "no, ship it", "no, go on", "well, no", "um, no, send it", "no, send it over",
    "hmm, no", "no no no",
])
def test_a_no_in_front_of_a_new_yes_head_is_still_a_no(said):
    assert parse_send_answer(said) is False, said


@pytest.mark.parametrize("said", [
    # a filler alone is not a yes
    "okay", "sure", "right", "well", "um", "okay okay", "hmm",
    # a name / an address / a second command after a new head is not a yes
    "ship it to Dana", "go on and turn the lights off", "send it over to Dana",
    "send it off to dana@example.com", "go for it and turn the lights off",
    "yes yes, and turn the lights off", "um yes send it to Dana",
    # a self-pronoun after a new head is not a yes
    "send it over to me", "go on, send it to me", "send it off to us",
    # the bare pronouns are not heads, with or without a "to" in front:
    # "to her--" is what an early endpoint makes of "to her work address"
    "her", "it", "that", "this", "to her", "to him", "to it", "to that",
])
def test_the_new_heads_do_not_widen_past_the_guards(said):
    assert parse_send_answer(said) is not True, said


@pytest.mark.parametrize("said", [
    "to the same address", "to that address", "to her address", "that address",
    "her inbox", "the same one", "that one", "that person",
])
def test_a_destination_with_a_noun_on_it_is_the_draft_said_back(said):
    assert parse_send_answer(said) is True, said


@pytest.mark.parametrize("said", [
    "send it over to me", "go on, send it to me", "um yes, send it to me",
])
def test_a_redirect_to_himself_behind_a_new_head_still_answers_self(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    FakeSMTP.made.clear()
    res = cmd.handle(said, source="typed")
    assert res is not None and "not to you" in res.reply, (said, res)
    assert not any(getattr(c, "sent", None) for c in FakeSMTP.made), said
    assert cmd._pending_send is not None


@pytest.mark.parametrize("said", [
    "um yes send it to Dana", "yes yes, send it to Dana",
    "okay yes, send it to dana@example.com",
])
def test_a_correction_behind_a_filler_still_corrects(cmd, said):
    """"okay yes, send it to dana at example dot com" is seven words with
    "okay" first -- past parse_yes_no's line -- so the raw sentence was
    no yes at all and the correction inside it was re-asked away; the
    next bare yes then sent the file to the person he had corrected
    AWAY from. The correction is read off the cleaned sentence now."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled, (said, res)
    assert not FakeSMTP.made, said
    # a correction reads the file back again (or asks for the address):
    # nothing sent, a question still open, and Heather is not the To.
    assert cmd.question_open(), (said, res.reply)
    cmd.handle("yes", source="typed")
    assert not (FakeSMTP.made and FakeSMTP.made[-1].sent
                and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"), said


def test_an_account_correction_behind_a_filler_still_corrects(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("well yes, but from my work account", source="typed")
    assert res is not None and res.reply.endswith("Send it, sir?"), res
    assert "your work account" in res.reply and not FakeSMTP.made
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"
    assert FakeSMTP.made[-1].logged_in[0] == "hunter@work.com"


@pytest.mark.parametrize("said", ["sure", "okay", "ok", "right", "fine"])
def test_a_filler_alone_still_gets_exactly_one_re_ask(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res.reply == outbox.unsure_line(cmd._pending_send), said
    assert not FakeSMTP.made and cmd._pending_send is not None


# ---- never silence while a draft is pending ----------------------------
def test_the_second_vague_answer_is_a_spoken_drop_not_silence(cmd):
    """It used to fall to the silent branch: the second "sure" went to the
    chat model and the draft died without a word."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("okay", source="typed")
    res = cmd.handle("sure", source="typed")
    assert res is not None and res.handled and res.speak, res
    assert res.reply == outbox.ASK_SPENT_LINE
    assert cmd._pending_send is None and not FakeSMTP.made
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


@pytest.mark.parametrize("said", ["the pdf", "that one there", "hmm let me think",
                                  "the one on the desktop", "mm the thing"])
def test_an_unrecognised_short_sentence_gets_the_re_ask_then_the_spoken_drop(cmd, said):
    """Not yes, not no, not a correction, and nothing below the read-back
    recognises it (this harness's router is a stub, so nothing does). The
    old handler returned None here: draft gone, no reply. Now: the one
    re-ask, and a second miss spends the draft out loud."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled and res.speak, (said, res)
    assert res.reply == outbox.unsure_line(cmd._pending_send), (said, res.reply)
    assert cmd._pending_send is not None and cmd._pending_send.reasked
    assert not FakeSMTP.made
    res = cmd.handle(said, source="typed")
    assert res is not None and res.reply == outbox.ASK_SPENT_LINE, (said, res)
    assert cmd._pending_send is None and cmd._send_aside is None
    assert not FakeSMTP.made
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made, "a yes after the spoken drop reaches nothing"


def test_the_re_ask_after_an_unrecognised_sentence_still_takes_a_yes(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("the pdf", source="typed")
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_the_ten_word_stray_is_a_spoken_drop_and_is_not_re_armed(cmd):
    """The sentence the whole grammar exists for. It must not re-arm the
    draft for the next stray "yeah" -- and it must not kill it in silence
    either. Long and unrecognised: the spoken drop, first time."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("Yeah, so you should be able to look that up.",
                     source="typed")
    assert res is not None and res.reply == outbox.ASK_SPENT_LINE, res
    assert cmd._pending_send is None and not FakeSMTP.made
    cmd.handle("yeah", source="typed")
    assert not FakeSMTP.made


def test_a_recognised_new_command_still_keeps_its_meaning_and_drops_the_draft(cmd):
    """The other half of the rule: a real command below the read-back is
    not re-asked, it is obeyed, and the draft is spent as before."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("what time is it", source="typed")
    assert res is not None and res.handled
    assert res.reply != outbox.ASK_SPENT_LINE and "rather be certain" not in str(res.reply)
    assert cmd._pending_send is None and cmd._send_aside is None
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


def test_a_routed_command_the_router_recognises_keeps_its_meaning(cmd):
    """With a real decision from the router (a strong local cue) the
    sentence is a command, not a miss: no re-ask, draft dropped."""
    from jarvis.router import RouteDecision
    cmd.services.router.route.return_value = RouteDecision("local", "local:weather")
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("what's the weather like in Houston", source="typed")
    assert res is not None and res.handled
    assert "rather be certain" not in str(res.reply) and res.reply != outbox.ASK_SPENT_LINE
    assert cmd._pending_send is None and not FakeSMTP.made


@pytest.mark.parametrize("intent", [IntentClassifier.NO, IntentClassifier.UNCERTAIN])
def test_the_intent_gate_cannot_drop_a_pending_draft_in_silence(cmd, monkeypatch, intent):
    """Voice, no address: the classifier calls four words background chat
    (NO) or asks "Was that for me?" (UNCERTAIN). With a draft pending both
    are the silent death; the draft answers first."""
    monkeypatch.setattr(cmd.intent, "classify", lambda text: (intent, 0.9))
    cmd.handle("email the biosensors handout to Heather", source="voice")
    assert cmd._pending_send is not None
    res = cmd.handle("the pdf", source="voice")
    assert res is not None and res.reply == outbox.unsure_line(cmd._pending_send), res
    assert cmd._pending_send is not None and not FakeSMTP.made
    res = cmd.handle("yes", source="voice")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_route_recognised_reads_the_decision_not_the_stub():
    from unittest.mock import MagicMock
    from jarvis.commander import _route_recognised
    from jarvis.router import RouteDecision
    assert not _route_recognised(MagicMock())
    assert not _route_recognised(None)
    for kind, reason in (("local", "short"), ("local", "classify"),
                         ("local", "empty"), ("local", "router-error"),
                         ("local", "local:topic")):
        assert not _route_recognised(RouteDecision(kind, reason)), (kind, reason)
    for kind, reason in (("local", "local:weather"), ("local", "local:question"),
                         ("local", "local:wrapper"), ("web", "web-cue"),
                         ("claude", "explicit"), ("action", "cancel"),
                         ("ask", "tie")):
        assert _route_recognised(RouteDecision(kind, reason)), (kind, reason)


def test_the_set_aside_draft_never_outlives_its_turn(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("Yeah, so you should be able to look that up.", source="typed")
    assert cmd._send_aside is None
    cmd.handle("the pdf", source="typed")
    assert cmd._send_aside is None


# ==================================================================
# 22. The fourth review (09-04): three bars on a yes, the pending name
#     said back, and "Which file, sir?" with a slot behind it
# ==================================================================
# Eight non-confirmations SENT to heather@example.com on the 0f1efe8
# grammar, every one through one of three holes: a trailing "?" was taken
# as yes-punctuation; "and" / "then" let a SECOND bare-pronoun recipient
# ride through, or ended the sentence; and "the" + a generic noun counted
# as the read-back echoed. None of these is a no either: each is asked
# again, out loud, with the file and the recipient named.
FOURTH_REVIEW_NEAR_YESES = [
    # a question is not an answer (Whisper writes "?" on a rising tone)
    "send it to her?", "yes?", "go on?", "to her address?", "send it through?",
    "yes, send it to her?", "the usual one?",
    # a second recipient on a bare pronoun after a connector
    "send it to her and to him", "send it to her, then him", "yes and him",
    "send it to her and to them", "yes, send it to her and it",
    "send it to heather and to him",
    # a sentence he was interrupted in
    "yes, and", "yes and", "send it to her and", "go on, and",
    # a bare determiner names nobody in particular
    "send it to the person", "send it to the one", "send it to the email",
    "send it to the address", "the person", "the one", "yes, to the address",
    # a correction that names nobody (were "I've no address for the, sir")
    "yes, one to her and one to Dana", "yes send it to the",
]


@pytest.mark.parametrize("said", FOURTH_REVIEW_NEAR_YESES)
def test_the_fourth_review_holes_are_not_a_yes(said):
    assert parse_send_answer(said) is not True, said


@pytest.mark.parametrize("said", FOURTH_REVIEW_NEAR_YESES)
def test_the_fourth_review_holes_are_asked_again_and_never_sent(cmd, said):
    """Not sent, not dropped, not silent: the one re-ask, draft kept, and
    the yes he gives to THAT sentence is the one that sends."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.speak, (said, res)
    assert cmd._pending_send is not None, (said, res.reply)
    assert res.reply == outbox.unsure_line(cmd._pending_send), (said, res.reply)
    assert "no address" not in res.reply.lower(), (said, res.reply)
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_a_question_mark_bars_only_the_yes():
    """A no with a question mark on it is still a no: nothing leaves the
    machine either way, and "wait, send it to her?" is a wait."""
    assert parse_send_answer("wait, send it to her?") is False
    assert parse_send_answer("no?") is False
    from jarvis.commander import _send_near_yes
    assert _send_near_yes("send it to her?") and _send_near_yes("yes, and")
    assert not _send_near_yes("no, send it to her?")
    assert not _send_near_yes("what's the weather")


@pytest.mark.parametrize("said", [
    # a ref WITH a noun on it may follow a connector: still the same person
    "yes, and to her address", "yes, then that one", "yes and that's the one",
    # "then" trailing is not a connector
    "do it then", "go ahead then", "send it on then", "yes then",
])
def test_a_connector_followed_by_the_same_person_is_still_a_yes(said):
    assert parse_send_answer(said) is True, said


# The twenty-two everyday yeses attack 2 measured as misses (19 re-asked,
# one re-read, two that killed the draft), plus their near neighbours.
FOURTH_REVIEW_SENDS = [
    # "through" beside over / along / off / out / on
    "send it through", "push it through", "send it through to her",
    "yes, send it through please", "send that through",
    # "then" rides on the lead filler
    "okay then, send it", "right then, off you go", "well then, yes",
    "so then, yes", "alright then, go on",
    # hyphenated backchannels and stutters
    "mm-hmm, yes", "uh-huh, send it", "y-yes", "s-send it", "ye- yes, send it",
    "mm-hmm, send it to her", "uh-huh, go on",
    # heads the grammar lacked
    "let's do it", "carry on", "that's fine, send it", "that'll do, send it",
    "that's fine", "that'll do", "yes, let's do it", "lets do it",
    # a truncated tail, and the usual without its noun
    "yes go", "yes send", "the usual", "yes, the usual",
]


@pytest.mark.parametrize("said", FOURTH_REVIEW_SENDS)
def test_the_fourth_review_yeses_are_a_yes_to_the_grammar(said):
    assert parse_send_answer(said) is True, said


@pytest.mark.parametrize("said", FOURTH_REVIEW_SENDS)
def test_the_fourth_review_yeses_send_to_the_pending_address(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled and res.ack, (said, res)
    assert "rather be certain" not in str(res.reply), (said, res.reply)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said
    assert cmd._pending_send is None and cmd._send_aside is None


@pytest.mark.parametrize("said,want", [
    ("okay then, send it", "send it"), ("right then, off you go", "off you go"),
    ("well then, yes", "yes"), ("mm-hmm, yes", "yes"), ("uh-huh, send it", "send it"),
    ("y-yes", "yes"), ("s-send it", "send it"), ("ye- yes, send it", "yes, send it"),
    # a real hyphenated word is not a stutter
    ("e-mail it to her", "e-mail it to her"),
    # "then" alone, or a filler and "then", is handed back whole
    ("okay then", "okay then"),
])
def test_what_send_clean_takes_off_now(said, want):
    from jarvis.commander import _send_clean
    assert _send_clean(said) == want


@pytest.mark.parametrize("said", [
    "no, send it through", "no, let's do it", "no, carry on", "no, that'll do",
    "no, that's fine", "well then, no", "mm-hmm, no", "n-no",
])
def test_a_no_still_beats_every_new_head(said):
    assert parse_send_answer(said) is False, said


@pytest.mark.parametrize("said", [
    "send it through to Dana", "push it through to dana@example.com",
    "let's do it and turn the lights off", "carry on with the lights",
    "that'll do, send it to Dana", "send it through to me", "yes go to Dana",
    "the usual, and turn the lights off", "okay then", "then",
])
def test_the_new_heads_still_stop_at_the_guards(said):
    assert parse_send_answer(said) is not True, said


# ---- the pending recipient's own name is the draft said back -----------
@pytest.mark.parametrize("said", [
    "yes, to Heather", "send it to heather", "yes, to heather's address",
    "yes, send it to Heather", "send it to heather please", "yes to heather's email",
    "send it to heather's inbox", "yes, send it to heather@example.com",
    "go on, send it to Heather",
])
def test_the_pending_name_after_to_is_a_yes_not_a_correction(cmd, said):
    """"yes, to Heather" was a correction to the person he had just been
    read (a needless re-read); "yes, to heather's address" became a
    recipient called "heather's address"; and "send it to heather" fell
    past the read-back rung to the send-file family, which asked "Which
    file, sir?" with the draft gone. The name he was read is the draft
    said back."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled and res.ack, (said, res)
    assert "no address" not in str(res.reply).lower(), (said, res.reply)
    assert res.reply != outbox.WHICH_FILE_LINE, said
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said
    assert cmd._pending_send is None and cmd._pending_sendask is None


@pytest.mark.parametrize("said", [
    "yes, to Dana", "send it to Dana", "yes, to dana's address",
    "send it to heather's mother", "send it to heather, then Dana",
    "send it to heather and to him", "no, send it to heather",
    "yes, to heather's boss",
])
def test_a_different_name_or_a_relative_of_hers_still_never_sends(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.reply, (said, res)


def test_what_the_fold_does():
    """The sixth pass (09-04) changed WHAT the name folds to: a neutral
    demonstrative the grammar already takes as the draft said back ("the
    same person" / "that address"), never a pronoun -- a synthetic "her" would
    trip the gender check on a Mr, and the pending name's gender is not
    the fold's to guess."""
    from jarvis.commander import _send_fold_pending
    draft = types.SimpleNamespace(to_name="Heather", to_addr="heather@example.com")
    assert _send_fold_pending("yes, to Heather", draft) == "yes, to the same person"
    assert _send_fold_pending("yes, to heather's address", draft) == "yes, to that address"
    assert _send_fold_pending("send it to heather@example.com", draft) == "send it to that address"
    assert _send_fold_pending("send it to heather at example dot com", draft) == "send it to that address"
    assert _send_fold_pending("send it to heather's mother", draft) == "send it to heather's mother"
    assert _send_fold_pending("yes, to Dana", draft) == "yes, to Dana"
    # the bare name folds only beside a yes of its own; alone it is left
    # to the re-ask
    assert _send_fold_pending("Heather, yes", draft) == "the same person, yes"
    assert _send_fold_pending("yes, Heather", draft) == "yes, the same person"
    assert _send_fold_pending("Heather", draft) == "Heather"
    assert _send_fold_pending("Heather?", draft) == "Heather?"
    # a two-word name folds on its first word too
    two = types.SimpleNamespace(to_name="Heather Smith", to_addr="")
    assert _send_fold_pending("yes, to heather", two) == "yes, to the same person"
    assert _send_fold_pending("yes, to Heather Smith", two) == "yes, to the same person"
    # a nameless address-only draft folds the address alone
    bare = types.SimpleNamespace(to_name="", to_addr="dana@example.com")
    assert _send_fold_pending("yes, to dana@example.com", bare) == "yes, to that address"
    assert _send_fold_pending("yes, to dana", bare) == "yes, to dana"


@pytest.mark.parametrize("said", [
    # the pending NAME inside a DIFFERENT address, typed or spoken: the
    # fold used to make "to her@gmail.com" of it and the next yes went to
    # a fabricated address (attack 1, round 2)
    "yes, send it to heather@gmail.com", "yes, send it to heather at gmail dot com",
    "yes send it to heather.jones@example.com", "to heather@gmail.com",
    "send it to heather-jones@example.com", "yes, to heather at work dot com",
    # a fuller name, an "at" phrase and a hyphenated name are not the
    # pending person either
    "yes, send it to Heather Jones", "send it to Heather-Jones",
    "yes, send it to heather at work",
])
def test_the_fold_never_rewrites_the_name_inside_an_address(said):
    from jarvis.commander import _send_fold_pending
    draft = types.SimpleNamespace(to_name="Heather", to_addr="heather@example.com")
    assert _send_fold_pending(said, draft) == said, said
    two = types.SimpleNamespace(to_name="Heather Smith", to_addr="heather@example.com")
    assert _send_fold_pending(said, two) == said, said


@pytest.mark.parametrize("said", ["yes, one to her and one to Dana", "yes send it to the",
                                  "yes, to the", "yes, send it to and Dana"])
def test_a_correction_that_names_nobody_is_not_a_correction(said):
    from jarvis.commander import _send_correction, _send_correction_malformed
    assert _send_correction(said) is None, said
    assert _send_correction_malformed(said), said


def test_a_correction_that_names_someone_is_not_malformed():
    from jarvis.commander import _send_correction_malformed
    for said in ("yes, send it to Dana", "yes, to my brother", "yes but to her work address",
                 "yes, to dana@example.com"):
        assert not _send_correction_malformed(said), said


# ---- "Which file, sir?" arms a slot ------------------------------------
def test_which_file_arms_a_slot_and_the_next_yes_is_answered_aloud(cmd):
    """"email it to Heather" with no file to point at: the question was
    spoken with nothing behind it, question_open() said no, and the "yes"
    after it returned None. Now it holds the floor like the other two
    send questions: a yes gets the question once more, a file phrase runs
    the send with it, and the read-back follows."""
    res = cmd.handle("email it to Heather", source="typed")
    assert res.reply == outbox.WHICH_FILE_LINE
    assert cmd._pending_sendask is not None and cmd._pending_sendask.kind == "file"
    assert cmd.question_open()
    res = cmd.handle("yes", source="typed")
    assert res is not None and res.handled and res.speak, res
    assert res.reply == outbox.WHICH_FILE_LINE
    assert cmd._pending_sendask is not None
    res = cmd.handle("the biosensors handout", source="typed")
    assert res.reply.endswith("Send it, sir?") and "to Heather" in res.reply
    assert cmd._pending_sendask is None and cmd._pending_send is not None
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_which_file_asked_twice_lets_go_out_loud(cmd):
    cmd.handle("email it to Heather", source="typed")
    cmd.handle("yes", source="typed")
    res = cmd.handle("yes", source="typed")
    assert res is not None and res.reply == outbox.ASK_DROPPED_LINE
    assert cmd._pending_sendask is None
    res = cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


def test_a_no_to_which_file_lets_go_out_loud(cmd):
    cmd.handle("email it to Heather", source="typed")
    res = cmd.handle("never mind", source="typed")
    assert res is not None and res.reply == outbox.ASK_SPENT_LINE
    assert cmd._pending_sendask is None and not FakeSMTP.made


def test_the_file_answer_takes_his_phrasings(cmd):
    cmd.handle("email it to Heather", source="typed")
    res = cmd.handle("it's the biosensors handout, please", source="typed")
    assert res.reply.endswith("Send it, sir?"), res.reply
    assert "Biosensors Lab Handout v2.pdf" in res.reply


def test_a_long_sentence_naming_no_file_is_a_new_subject_to_which_file(cmd):
    cmd.handle("email it to Heather", source="typed")
    res = cmd.handle("what is the capital of france would you say", source="typed")
    assert res is None or res.reply != outbox.WHICH_FILE_LINE
    assert cmd._pending_sendask is None


# ==================================================================
# 23. The fifth pass (09-04): a second recipient of any kind is the
#     re-ask, "and then" dangles like "and", and Hunter's ruling
# ==================================================================
# Attack 1's seven-word second-recipient sentences ("send it to her and
# cc Dana", "... and 3 others", "yes, 2 copies to her and Dana", "... and
# to Dana") were dropped OUT LOUD on the aside path, and "send it to her
# and dana@example.com" fell to the send-file family and asked "Which
# file, sir?". He said "to her" and then named somebody else: the one
# re-ask, whatever the length, and the yes he gives to THAT sentence
# sends to her.
FIFTH_PASS_SECOND_RECIPIENTS = [
    "send it to her and to Dana", "send it to her and cc Dana", "send it to her cc Dana",
    "send it to her and 3 others", "yes, 2 copies to her and Dana",
    "send it to her and dana@example.com", "send it to her and Dana",
    "send it to her, then Dana", "send it to her and Dana please",
    "send it to her and her mother", "send it to her and the whole department",
    "yes, send it to her and to Dana", "send it to her and to heather",
    "send it to her and me", "send it to her, plus Dana", "send it to her and also Dana",
    "send it to her and copy Dana", "send it to her and one to Dana",
]


@pytest.mark.parametrize("said", FIFTH_PASS_SECOND_RECIPIENTS)
def test_a_second_recipient_of_any_kind_is_not_a_yes(said):
    assert parse_send_answer(said) is not True, said


@pytest.mark.parametrize("said", FIFTH_PASS_SECOND_RECIPIENTS)
def test_a_second_recipient_of_any_kind_is_the_re_ask(cmd, said):
    """Not sent, not dropped, not "Which file, sir?", not a correction to
    anybody: the one re-ask with the draft kept."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.speak, (said, res)
    assert cmd._pending_send is not None, (said, res.reply)
    assert res.reply == outbox.unsure_line(cmd._pending_send), (said, res.reply)
    assert res.reply not in (outbox.ASK_SPENT_LINE, outbox.WHICH_FILE_LINE), said
    assert "no address" not in res.reply.lower(), (said, res.reply)
    assert cmd._pending_sendask is None, said
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


def test_what_counts_as_a_second_recipient():
    from jarvis.commander import _send_near_yes, _send_second_recipient
    for said in ("send it to her and to Dana", "to her and cc Dana",
                 "send it to her and 3 others", "yes, 2 copies to her and Dana",
                 "send it to her and dana@example.com", "send it to her, then Dana please",
                 "send it to her address and Dana"):
        assert _send_second_recipient(said) and _send_near_yes(said), said
    # a command after the read-back's words keeps its meaning below this rung
    for said in ("send it to her and turn the lights off",
                 "send it to her and then turn the lights off",
                 "send it to her and what is the time now"):
        assert not _send_second_recipient(said), said
    # the same person with a noun on it, a thanks, a trailing then: a yes,
    # not a near one
    for said in ("yes, to her and to her address", "send it to her and thanks",
                 "send it to her then, thanks", "yes, then that one"):
        assert not _send_near_yes(said) and parse_send_answer(said) is True, said
    # ends on a connector: the bar, not this rule
    assert _send_near_yes("send it to her and")
    assert not _send_second_recipient("send it to her and")


@pytest.mark.parametrize("said", ["yes and then", "yes, and then", "send it to her and then",
                                  "go on and then", "yes then and"])
def test_and_then_dangles_like_and(said):
    assert parse_send_answer(said) is not True, said


def test_yes_and_then_is_asked_again_not_sent(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("yes and then", source="typed")
    assert not FakeSMTP.made
    assert res.reply == outbox.unsure_line(cmd._pending_send)
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


@pytest.mark.parametrize("said", ["do it then", "yes then", "go on then", "send it on then"])
def test_a_trailing_then_alone_is_still_a_yes(said):
    assert parse_send_answer(said) is True, said


# ---- Hunter's ruling (16:58, 09-04): after a read-back, "okay send it to
# her" SENDS -- "okay" is a filler and "send it to her" is the read-back's
# own words said back. A trailing "?" still re-asks.
HUNTERS_RULING_SENDS = [
    # ("okay, send it to him" moved to GENDER_REASKS, section 24: the
    # 19:00 ruling -- a pronoun has to match the pending person)
    "okay send it to her", "okay, send it to her", "alright send it to her",
    "right, send it to her", "sure, send it to her",
    "okay send it to heather", "okay, send it to her address",
]
HUNTERS_RULING_REASKS = [
    "send it to her?", "okay send it to her?", "okay, send it to her?",
    "alright send it to her?", "yes?", "go on?", "to her address?",
]


@pytest.mark.parametrize("said", HUNTERS_RULING_SENDS)
def test_hunters_ruling_a_filler_on_the_read_backs_own_words_sends(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled and res.ack, (said, res)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said
    assert cmd._pending_send is None


@pytest.mark.parametrize("said", HUNTERS_RULING_REASKS)
def test_hunters_ruling_a_question_mark_still_re_asks(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert cmd._pending_send is not None, (said, res)
    assert res.reply == outbox.unsure_line(cmd._pending_send), (said, res.reply)
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


# ==================================================================
# 24. The sixth pass (09-04, evening): the round-2 attacks, Hunter's
#     19:00 gender ruling, and the address-book hand-off
# ==================================================================
# Round 2's two attacks and the verdict, reproduced on this harness at
# 1e27530: 15 of 73 non-confirmations SENT (a "?" not at the very end, a
# second recipient joined by a comma / full stop / "now" / a second head,
# a connector run ending on a tail word), the "no, I said X" rung stripped
# the "?" off "I said yes?" and re-dispatched a bare yes, and the fold
# rewrote the pending NAME inside a DIFFERENT address so the next yes went
# to her@gmail.com -- an address he never said. Then Hunter's 19:00
# ruling: a pronoun has to match the pending person's gender, from an
# explicit source only. And three hand-offs from the address-book review.

# ---- (a) a "?" ANYWHERE in the cleaned sentence is never a yes ------------
SIXTH_PASS_QUESTIONS = [
    "send it to her?!", "yes?!", "send it to her? send it to him.", "to her? really?",
    "okay send it to him?", "yes, really?", "send it to her? yes.", "positive?",
    "go?", "please?", "let's go?", "that is correct?",
]
# ---- (b) a second recipient by ANY joiner ---------------------------------
SIXTH_PASS_SECOND_RECIPIENTS = [
    "send it to her, him", "send it to her, to him", "send it to her, then send it to him",
    "send it to her and send it to him", "send it to her, now send it to him",
    "yes, send it to her, send it to him", "send it to her. now him.", "send it to her. him.",
    "yes, him", "yes, Dana too", "send it to him and her", "yes, her and Dana",
    "send it to her, him, and Dana", "yes, him, not her", "send it to her or him",
    "send it to her, her mother too", "yes, them too",
    # a second recipient after the read-back's own words, by a whole "send
    # it to" or a "with a copy to" (were a correction to Dana / a spoken
    # drop on the fixed build's first replay)
    "okay send it to her, then send it to Dana", "send it to her with a copy to Dana",
    "send it to her, now send it to Dana", "yes, send it to her and send it to Dana",
]
# ---- (c) a trailing connector run ending on and / then / send / plus -----
SIXTH_PASS_DANGLES = [
    "yes and then send", "yes and then then", "yes then send", "yes and then go",
    "yes, but", "yes, or", "send it to her but", "yes and plus", "yes, plus",
    "send it to her and then and",
]


@pytest.mark.parametrize("said", SIXTH_PASS_QUESTIONS + SIXTH_PASS_SECOND_RECIPIENTS
                         + SIXTH_PASS_DANGLES)
def test_the_sixth_pass_holes_are_not_a_yes(said):
    assert parse_send_answer(said) is not True, said


@pytest.mark.parametrize("said", SIXTH_PASS_QUESTIONS + SIXTH_PASS_DANGLES)
def test_the_sixth_pass_holes_are_asked_again_and_never_sent(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.speak, (said, res)
    assert cmd._pending_send is not None, (said, res.reply)
    assert res.reply in (outbox.unsure_line(cmd._pending_send),
                         outbox.gender_line(cmd._pending_send)), (said, res.reply)
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


@pytest.mark.parametrize("said", SIXTH_PASS_SECOND_RECIPIENTS)
def test_a_second_recipient_by_any_joiner_is_the_re_ask(cmd, said):
    """The check runs on the WHOLE sentence, before the yes grammar can
    return True: a comma, a full stop, "now", "or", a second head -- any
    joiner between two people is the one re-ask, draft kept."""
    from jarvis.commander import _send_second_recipient
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.speak, (said, res)
    assert cmd._pending_send is not None, (said, res.reply)
    assert res.reply in (outbox.unsure_line(cmd._pending_send),
                         outbox.gender_line(cmd._pending_send)), (said, res.reply)
    assert "no address" not in res.reply.lower(), (said, res.reply)
    if "Dana" not in said:
        assert _send_second_recipient(said), said
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


@pytest.mark.parametrize("said", [
    # one person, said back with a noun on it, a thanks or a trailing then
    "yes, to her and to her address", "send it to her and thanks", "yes, send it to her, thanks",
    "send it to her then", "yes, to her, go ahead", "yes, her inbox",
])
def test_one_person_twice_is_not_a_second_recipient(said):
    from jarvis.commander import _send_second_recipient
    assert not _send_second_recipient(said), said
    assert parse_send_answer(said) is True, said


# ---- (d) the correction rung keeps a trailing "?" -------------------------
I_SAID_QUESTIONS = [
    "I said yes?", "I said, yes?", "no, I said yes?", "I said send it to her?",
    "I meant yes?", "I mean, send it to her?", "I said go ahead?",
    "what I said was yes?", "I was saying yes?",
]


@pytest.mark.parametrize("text,meant", [
    ("I said yes?", "yes?"), ("no, I said yes?", "yes?"), ("I said, yes?", "yes?"),
    ("I mean, send it to her?", "send it to her?"), ("no, I said yes.", "yes"),
    ("I said yes?.", "yes?"), ("not the terminal, the calendar?", "the calendar?"),
])
def test_correction_kind_keeps_the_question_mark(text, meant):
    from jarvis.commander import correction_kind
    assert correction_kind(text) == meant


@pytest.mark.parametrize("said", I_SAID_QUESTIONS)
def test_i_said_yes_with_a_question_mark_re_asks_and_never_sends(cmd, said):
    """The "no, I said X" rung re-dispatches X ahead of the send rung; it
    used to throw the "?" away first, so "I said yes?" SENT. Now "yes?"
    reaches the bar. Whisper writes "?" on a rising tone."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.speak, (said, res)
    assert cmd._pending_send is not None, (said, res.reply)
    assert res.reply == outbox.unsure_line(cmd._pending_send), (said, res.reply)
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"


@pytest.mark.parametrize("said", ["no, I said yes", "I said yes", "no, I said send it to her",
                                  "I meant yes", "I said go ahead"])
def test_no_i_said_yes_is_still_a_yes(cmd, said):
    """The 08-30 design, still pinned: a correction is not a decline."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.ack, (said, res)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said


# ---- (e) the fold and a DIFFERENT address of hers -------------------------
@pytest.mark.parametrize("said,to", [
    ("yes, send it to heather@gmail.com", "heather@gmail.com"),
    ("yes, send it to heather at gmail dot com", "heather@gmail.com"),
    ("yes send it to heather.jones@example.com", "heather.jones@example.com"),
    ("send it to heather@gmail.com", "heather@gmail.com"),
    ("no, to heather@gmail.com", "heather@gmail.com"),
    ("send it to heather at gmail dot com instead", "heather@gmail.com"),
])
def test_another_address_of_hers_is_read_back_as_said_never_mangled(cmd, said, to):
    """Attack 1's worst class: "yes, send it to heather@gmail.com" was read
    back "to her at gmail dot com" and the next yes SENT To=her@gmail.com.
    The address is parsed off the UNFOLDED sentence; the fold stays away
    from any sentence that holds an address."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.reply.endswith("Send it, sir?"), (said, res)
    assert outbox.spoken_address(to) in res.reply, (said, res.reply)
    assert " her at" not in res.reply and " her dot" not in res.reply, res.reply
    assert "that address" not in res.reply and "same person" not in res.reply, res.reply
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == to, said


@pytest.mark.parametrize("pending,book", [
    ("Heather", {"ms heather": "heather@example.com"}),
    ("Heather Smith", {"ms heather smith": "heather@example.com"}),
])
def test_a_fuller_name_is_a_correction_to_that_whole_name(cmd, pending, book):
    """"yes, send it to Heather Jones" with Heather (or Heather Smith)
    pending is a correction to HEATHER JONES -- not to "Jones", and not the
    draft said back."""
    cmd.services.assistant.data["send_file.contacts"] = book
    res = cmd.handle(f"email the biosensors handout to {pending}", source="typed")
    assert res.reply.endswith("Send it, sir?") and pending in res.reply, res.reply
    res = cmd.handle("yes, send it to Heather Jones", source="typed")
    assert not FakeSMTP.made
    assert res.reply == outbox.NO_RECIPIENT_LINE.format(who="Heather Jones"), res.reply
    assert cmd._pending_send is None
    assert cmd._pending_sendask is not None and cmd._pending_sendask.kind == "recipient"
    assert cmd.question_open()


# ---- Hunter's 19:00 ruling: a pronoun must match the pending person ------
# GENDER SOURCE, explicit only: a stored honorific on the person or the book
# row (Mr / Mrs / Ms / Miss / Sir / Madam), or a pronoun Hunter himself used
# about the person earlier in the same draft conversation. NO name-based
# guessing: a bare "heather" row says nothing, and then either pronoun
# confirms exactly as before. ONE seam: Draft.to_gender, None by default,
# filled by whoever knows (outbox.prepare from the book / people book, the
# address answer from his own pronoun), so the address-book branch can fill
# it from its rows later without touching the grammar.
GENDER_LINE_HEATHER = "The draft is to Heather, sir. Send it to her?"
GENDER_REASKS = [
    # the five the brief moved out of sections 18 / 20 / 23 ...
    "okay send it to him", "send it to him", "yes, to him", "yeah, send it to him",
    "send it to him please",
    # ... and the pinned sends that carried a "him" / "his" with them
    "yes send it to him", "yes send it to him please", "okay, send it to him",
    "yes, send it to his inbox",
    # near neighbours
    "yes, send it to his address", "send it to him, jarvis", "go on, send it to him",
    "please send it to him", "yes go ahead and send it to him",
]


@pytest.mark.parametrize("said", GENDER_REASKS)
def test_a_pronoun_of_the_other_gender_is_not_a_confirmation(cmd, said):
    """Heather's row says Ms. "him" is not her: the re-ask names the pending
    person and her pronoun, nothing is sent, the draft is kept, and the yes
    he gives to THAT sentence sends."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.speak, (said, res)
    assert cmd._pending_send is not None, (said, res.reply)
    assert res.reply == GENDER_LINE_HEATHER, (said, res.reply)
    assert res.reply == outbox.gender_line(cmd._pending_send)
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said


@pytest.mark.parametrize("said", GENDER_REASKS)
def test_the_grammar_itself_stays_gender_blind(said):
    """The gender lives on the DRAFT, not in the grammar: parse_send_answer
    still takes "send it to him" as the shape of a yes."""
    assert parse_send_answer(said) is True, said


def test_after_the_gender_re_ask_her_pronoun_sends_and_his_spends_it_aloud(cmd):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("send it to him", source="typed")
    res = cmd.handle("send it to her", source="typed")
    assert res.ack and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com"
    FakeSMTP.made.clear()
    cmd.handle("email the biosensors handout to Heather", source="typed")
    cmd.handle("send it to him", source="typed")
    res = cmd.handle("send it to him", source="typed")
    assert res.reply == outbox.ASK_SPENT_LINE and not FakeSMTP.made
    assert cmd._pending_send is None
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


@pytest.mark.parametrize("said", ["send it to him", "okay send it to him", "yes, to him",
                                  "send it to her", "yes send it to his inbox"])
def test_an_unknown_gender_takes_either_pronoun_exactly_as_before(cmd_plain, said):
    cmd = cmd_plain
    cmd.handle("email the biosensors handout to Heather", source="typed")
    assert cmd._pending_send is not None and cmd._pending_send.to_gender is None
    res = cmd.handle(said, source="typed")
    assert res is not None and res.ack, (said, res)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said


def test_a_mr_on_the_row_makes_her_the_wrong_pronoun(cmd):
    cmd.services.assistant.data["send_file.contacts"]["mr jones"] = "jones@example.com"
    res = cmd.handle("email the biosensors handout to Mr Jones", source="typed")
    assert res.reply.endswith("Send it, sir?") and "to Mr Jones" in res.reply, res.reply
    assert cmd._pending_send.to_gender == "m"
    res = cmd.handle("send it to her", source="typed")
    assert not FakeSMTP.made
    assert res.reply == "The draft is to Mr Jones, sir. Send it to him?", res.reply
    res = cmd.handle("send it to him", source="typed")
    assert res.ack and FakeSMTP.made[-1].sent[0]["To"] == "jones@example.com"


def test_the_honorific_on_the_row_counts_when_the_name_is_said_bare(cmd):
    """The row says "mr jones"; he says "Jones". The row still resolves
    through its honorific-stripped key, and the honorific is the gender."""
    cmd.services.assistant.data["send_file.contacts"]["mr jones"] = "jones@example.com"
    res = cmd.handle("email the biosensors handout to Jones", source="typed")
    assert res.reply.endswith("Send it, sir?") and "to Jones, at jones at" in res.reply, res.reply
    assert cmd._pending_send.to_gender == "m"
    res = cmd.handle("yes, send it to her address", source="typed")
    assert not FakeSMTP.made
    assert res.reply == "The draft is to Jones, sir. Send it to him?", res.reply


def test_the_people_book_honorific_fills_the_seam(cmd):
    cmd.services.memory.resolve_person.return_value = {
        "name": "Dana", "email": "dana@example.com", "honorific": "Mrs"}
    res = cmd.handle("email the biosensors handout to Dana", source="typed")
    assert res.reply.endswith("Send it, sir?") and "to Dana" in res.reply, res.reply
    assert cmd._pending_send.to_gender == "f"
    res = cmd.handle("send it to him", source="typed")
    assert not FakeSMTP.made
    assert res.reply == "The draft is to Dana, sir. Send it to her?", res.reply


def test_his_own_pronoun_in_the_draft_conversation_fills_the_seam(cmd):
    """"I've no address for Dana, sir. What is it?" -- "her address is dana
    at example dot com". He called Dana "her": that is the pronoun the
    confirmation has to match."""
    res = cmd.handle("email the biosensors handout to Dana", source="typed")
    assert res.reply == outbox.NO_RECIPIENT_LINE.format(who="Dana")
    res = cmd.handle("her address is dana at example dot com", source="typed")
    assert res.reply.endswith("Send it, sir?") and "to Dana, at dana at" in res.reply, res.reply
    assert cmd._pending_send.to_gender == "f"
    res = cmd.handle("send it to him", source="typed")
    assert not FakeSMTP.made
    assert res.reply == "The draft is to Dana, sir. Send it to her?", res.reply
    res = cmd.handle("yes, send it to her", source="typed")
    assert res.ack and FakeSMTP.made[-1].sent[0]["To"] == "dana@example.com"


def test_the_seam_and_its_explicit_sources():
    from jarvis.outbox import Draft, gender_from_honorific, gender_from_pronouns
    draft = Draft(path=Path("/tmp/x.pdf"), size=1, mtime=0.0, to_addr="a@b.co",
                  to_name="Heather", account={}, subject="s")
    assert draft.to_gender is None
    assert gender_from_honorific("Mrs Jones") == "f"
    assert gender_from_honorific("Ms Heather") == "f"
    assert gender_from_honorific("Miss Heather Smith") == "f"
    assert gender_from_honorific("Madam Secretary") == "f"
    assert gender_from_honorific("Mr Jones") == "m"
    assert gender_from_honorific("Mr. Jones") == "m"
    assert gender_from_honorific("Sir Isaac") == "m"
    assert gender_from_honorific("Dr Jones") is None
    assert gender_from_honorific("Heather") is None          # no guessing from a name
    assert gender_from_honorific("Dana") is None
    assert gender_from_honorific("James") is None
    assert gender_from_honorific("f") == "f" and gender_from_honorific("male") == "m"
    assert gender_from_honorific("") is None
    assert gender_from_pronouns("her address is dana at example dot com") == "f"
    assert gender_from_pronouns("he's at jones at example dot com") == "m"
    assert gender_from_pronouns("his address is x at y dot com") == "m"
    assert gender_from_pronouns("dana at example dot com") is None
    assert gender_from_pronouns("her and his") is None       # both: nobody's
    assert gender_from_pronouns("") is None
    assert outbox.gender_line(draft) == "The draft is to Heather, sir. Send it to them?"
    draft.to_gender = "f"
    assert outbox.gender_line(draft) == "The draft is to Heather, sir. Send it to her?"


# ---- hand-off (f): SELF_LINE speaks the address --------------------------
def test_self_line_speaks_a_typed_address(cmd):
    res = cmd.handle("email the biosensors handout to dana@example.com", source="typed")
    assert res.reply.endswith("Send it, sir?") and "dana at example dot com" in res.reply
    res = cmd.handle("send it to me", source="typed")
    assert not FakeSMTP.made and cmd._pending_send is not None
    assert "not to you" in res.reply and "dana at example dot com" in res.reply, res.reply
    assert "@" not in res.reply, res.reply


# ---- hand-off (g): a correction without a yes on it -----------------------
@pytest.mark.parametrize("said", [
    "no, to Heather Jones", "no, send it to Heather Jones", "send it to Heather Jones instead",
    "to Heather Jones", "send it to Heather Jones", "okay, to Heather Jones",
])
def test_a_correction_without_a_yes_is_still_a_correction(cmd, said):
    """Pre-existing on jarvis-v3: these were a SILENT DROP after a
    read-back, and the "yes" after them sent nothing. A name after "to" is
    a correction whether or not a yes rides in front of it: the unknown
    name gets the address question with a slot behind it."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.speak, (said, res)
    assert res.reply == outbox.NO_RECIPIENT_LINE.format(who="Heather Jones"), (said, res.reply)
    assert cmd._pending_send is None
    assert cmd._pending_sendask is not None and cmd._pending_sendask.kind == "recipient"
    res = cmd.handle("yes", source="typed")
    assert res is not None and res.reply == outbox.ADDRESS_REASK_LINE, (said, res)
    assert not FakeSMTP.made


@pytest.mark.parametrize("said", [
    "no, to Dana", "no, send it to Dana", "send it to Dana instead", "to Dana",
    "send it to Dana",
])
def test_a_correction_without_a_yes_to_a_known_name_is_read_back(cmd, said):
    cmd.services.assistant.data["send_file.contacts"]["dana"] = "dana@example.com"
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.reply.endswith("Send it, sir?"), (said, res)
    assert "dana at example dot com" in res.reply and "heather" not in res.reply.lower()
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "dana@example.com", said


@pytest.mark.parametrize("said", ["no, to Heather", "no, send it to heather", "not to her",
                                  "no, to her"])
def test_a_no_with_the_pending_person_on_it_never_sends(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made and res is not None and res.reply, (said, res)
    assert "no address" not in res.reply.lower(), (said, res.reply)
    cmd.handle("yes", source="typed")
    assert not (FakeSMTP.made and FakeSMTP.made[-1].sent
                and FakeSMTP.made[-1].sent[0]["To"] != "heather@example.com")


# ---- hand-off (h): the INFO lines carry no address ------------------------
def test_the_confirm_paths_log_no_address(cmd, caplog):
    caplog.set_level(logging.INFO)
    for said in ("yes, send it to dana@example.com", "send it to dana@example.com",
                 "yes, send it to dana at example dot com", "no, to dana@example.com",
                 "dana@example.com", "no, I said send it to dana@example.com",
                 "yes, send it to her and dana@example.com", "send it to me, dana@example.com",
                 "dana@example.com please right now if you would be so kind"):
        cmd.handle("email the biosensors handout to Heather", source="typed")
        cmd.handle(said, source="typed")
        cmd.handle("never mind", source="typed")
    cmd.handle("email the biosensors handout to Dana", source="typed")
    cmd.handle("dana at example dot com", source="typed")
    cmd.handle("no", source="typed")
    cmd.handle("email the biosensors handout to dana@example.com", source="typed")
    cmd.handle("no", source="typed")
    assert caplog.text, "nothing was logged at INFO -- the probe proves nothing"
    assert "dana@example.com" not in caplog.text
    assert "dana at example dot com" not in caplog.text
    assert "heather@example.com" not in caplog.text
    assert not FakeSMTP.made


def test_mask_addresses():
    assert outbox.mask_addresses("yes, to dana@example.com now") == "yes, to d…@example.com now"
    # The address-book lane's mask (the superset) keeps the first letter,
    # as mail._mask_address does for a typed one.
    assert outbox.mask_addresses("to dana at example dot com") == "to d… at example dot com"
    assert outbox.mask_addresses("send it to her") == "send it to her"


# ---- the missed yeses --------------------------------------------------
SIXTH_PASS_SENDS = [
    "just do it", "just send it", "let's go", "send it already", "send it straight away",
    "ship it over", "that is correct", "yes, that's it", "yes, exactly", "yes, I'm sure",
    "positive", "yes, really", "yes, I'm certain", "yes send it, I'm sure",
    "yes, go ahead, I'm sure", "yes, definitely",
    # an approval word in front of a head
    "sounds good, send it", "perfect, send it", "sure thing, send it", "righto, send it",
    "no worries, send it", "ya, send it", "good, send it", "great, send it",
    # bare "please" and "go" after a read-back
    "please", "go",
    # a three-letter stutter
    "sen- send it", "sen-send it to her",
]
SIXTH_PASS_NAME_SENDS = [
    "to Heather, yes", "yes, Heather", "Heather, yes, send it", "to Heather please",
    "yes, to Heather, go ahead", "to Heather", "Heather, send it", "yes to Heather",
]


@pytest.mark.parametrize("said", SIXTH_PASS_SENDS)
def test_the_sixth_pass_yeses_are_a_yes_to_the_grammar(said):
    assert parse_send_answer(said) is True, said


@pytest.mark.parametrize("said", SIXTH_PASS_SENDS + SIXTH_PASS_NAME_SENDS)
def test_the_sixth_pass_yeses_send_to_the_pending_address(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled and res.ack, (said, res)
    assert "rather be certain" not in str(res.reply), (said, res.reply)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said
    assert cmd._pending_send is None and cmd._send_aside is None


@pytest.mark.parametrize("said", SIXTH_PASS_SENDS + SIXTH_PASS_NAME_SENDS + [
    "yes, I'm sure", "yes, really", "yes, I'm certain", "yes send it, I'm sure", "positive",
    "yes, go ahead, I'm sure", "yes", "send it to her", "just send it", "please",
])
def test_on_the_second_turn_any_yes_the_grammar_takes_sends(cmd, said):
    """After the one re-ask, the reassurance he gives ("yes, I'm sure")
    used to be spent aloud -- two yeses and nothing sent."""
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle("okay", source="typed")
    assert res.reply == outbox.unsure_line(cmd._pending_send) and cmd._pending_send.reasked
    res = cmd.handle(said, source="typed")
    assert res is not None and res.ack, (said, res)
    assert res.reply != outbox.ASK_SPENT_LINE, (said, res.reply)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather@example.com", said


@pytest.mark.parametrize("said,want", [
    ("sen- send it", "send it"), ("sen-send it to her", "send it to her"),
    ("just send it", "send it"), ("yes, just send it", "yes, send it"),
    ("perfect, send it", "send it"), ("sure thing, send it", "send it"),
    ("no worries, send it", "send it"), ("sounds good, send it", "send it"),
    ("righto, send it", "send it"), ("exactly, send it", "send it"),
    # not a stutter, not a filler
    ("e-mail it to her", "e-mail it to her"), ("re-send it", "re-send it"),
    ("co-op", "co-op"),
    ("justin, send it", "justin, send it"), ("no worries", "no worries"),
])
def test_what_send_clean_takes_off_in_the_sixth_pass(said, want):
    from jarvis.commander import _send_clean
    assert _send_clean(said) == want


# ---- the guards beside the new heads -------------------------------------
@pytest.mark.parametrize("said", [
    "go to Dana", "go away", "let's go to Dana", "ship it over to Dana", "please, to Dana",
    "just send it to Dana", "positive?", "yes, exactly, and turn the lights off", "really",
    "sure thing", "no worries", "sounds good", "perfect", "good", "exactly", "exactly?",
    "go?", "please don't", "please stop", "yes, I'm sure, to Dana",
    "let's go and turn the lights off", "just", "just to Dana", "ship it over to me",
    "go on and send it to me", "please send it to me",
])
def test_the_sixth_pass_heads_stop_at_the_guards(said):
    assert parse_send_answer(said) is not True, said


@pytest.mark.parametrize("said", ["send it to Heather's", "yes, to her boss's",
                                  "send it to heather's", "yes, send it to Dana's"])
def test_a_bare_possessive_is_the_re_ask_not_an_address_for_bosss(cmd, said):
    """"send it to Heather's" -- the noun fell off an early endpoint. Not
    a correction to a person called "Heather's": the re-ask."""
    from jarvis.commander import _send_correction, _send_correction_malformed
    assert _send_correction(said) is None, said
    assert _send_correction_malformed(said), said
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res.reply == outbox.unsure_line(cmd._pending_send), (said, res.reply)
    assert cmd._pending_sendask is None


@pytest.mark.parametrize("said", ["go ahead and send it to him too", "yes, I told him I'd send it",
                                  "okay send it to him when he's back"])
def test_a_wrong_pronoun_in_a_sentence_nothing_takes_is_the_re_ask_not_a_drop(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert cmd._pending_send is not None, (said, res.reply)
    assert res.reply in (outbox.unsure_line(cmd._pending_send),
                         outbox.gender_line(cmd._pending_send)), (said, res.reply)


@pytest.mark.parametrize("said", ["Heather", "Heather?", "Dana, yes, send it", "yes, Dana",
                                  "Heather Jones, yes", "yes, Heather Jones"])
def test_a_bare_name_or_another_name_beside_a_yes_never_sends(cmd, said):
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.handled and res.reply, (said, res)
    assert cmd._pending_send is not None or cmd._pending_sendask is not None, said


# ---- a late no is a no ----------------------------------------------------
@pytest.mark.parametrize("said", [
    "please don't", "send it to her, actually no", "send it to her. no.", "yes, hold on",
    "yes, send it, no wait", "yeah go ahead, actually no", "yep, send it, hang on",
    "yes, on second thought don't", "please, no", "yes, wait",
])
def test_a_late_no_is_a_no(cmd, said):
    assert parse_send_answer(said) is False, said
    cmd.handle("email the biosensors handout to Heather", source="typed")
    res = cmd.handle(said, source="typed")
    assert not FakeSMTP.made and res.reply == outbox.DROPPED_LINE, (said, res)
    assert cmd._pending_send is None
    cmd.handle("yes", source="typed")
    assert not FakeSMTP.made


@pytest.mark.parametrize("said", ["no worries, send it", "yes, no doubt"])
def test_a_no_word_that_is_not_a_no_still_sends(said):
    assert parse_send_answer(said) is True, said


# ==================================================================
# 25. Round 4 (09-05): a spoken address with a joiner in its local part
# is read WHOLE or asked again -- never cut to its tail and sent there
# ==================================================================
# Attack 1 reproduced one breach class, typed == voice, 16 rows: "yes,
# send it to heather underscore smith at example dot com" was read back
# "to smith at example dot com" and the next yes SENT To=smith@example.com.
# The parser knew "dot" as the only spoken joiner, so the match began at
# the last word before "at" and the joiner and everything in front of it
# were dropped in silence. Worse, spoken_address() itself speaks "_" as
# "underscore", so Jarvis's OWN read-back of heather_smith@example.com,
# said back to it word for word, went to a stranger.

# ---- (a) the parser reads every joiner the speaker speaks ---------------
@pytest.mark.parametrize("addr", [
    "heather_smith@example.com", "heather-smith@example.com",
    "h.peyrovi@tamu.edu", "h_p-q.r@my-host.example.com",
    "heather+lab@example.com",
])
def test_a_spoken_address_round_trips(addr):
    """parse_address(spoken_address(x)) == x for a local part with _ - . +
    -- the read-back's own words are an address the parser reads."""
    assert outbox.parse_address(outbox.spoken_address(addr)) == addr


@pytest.mark.parametrize("said,want", [
    ("heather underscore smith at example dot com", "heather_smith@example.com"),
    ("heather under score smith at example dot com", "heather_smith@example.com"),
    ("heather dash smith at example dot com", "heather-smith@example.com"),
    ("heather hyphen smith at example dot com", "heather-smith@example.com"),
    ("h underscore peyrovi at tamu dot edu", "h_peyrovi@tamu.edu"),
    ("h period peyrovi at tamu dot edu", "h.peyrovi@tamu.edu"),
    ("h dot peyrovi at tamu dot edu", "h.peyrovi@tamu.edu"),
    ("heather at my dash host dot com", "heather@my-host.com"),
    ("yes, send it to heather underscore smith at example dot com", "heather_smith@example.com"),
])
def test_a_joiner_inside_the_local_part_is_read(said, want):
    assert outbox.parse_address(said) == want, said


# ---- (b) a local part the parser cannot read is never cut to its tail ---
@pytest.mark.parametrize("said,heard", [
    ("heather tilde smith at example dot com", "heather tilde smith at example dot com"),
    ("yes, send it to heather slash smith at example dot com",
     "heather slash smith at example dot com"),
    ("h star heather tilde smith at example dot com",
     "h star heather tilde smith at example dot com"),
    # "plus" is the second-recipient connector ("send it to her, plus
    # Dana"); read as "+" it makes an address out of two people, so it is
    # heard and handed back, never resolved.
    ("heather plus lab at example dot com", "heather plus lab at example dot com"),
    ("yes, send it to her plus dana at example dot com",
     "her plus dana at example dot com"),
    # typed, with a character in front an address cannot carry
    ("send it to heather~smith@example.com", "heather~smith at example dot com"),
])
def test_an_unreadable_local_part_is_no_address_and_is_heard_whole(said, heard):
    assert outbox.parse_address(said) == "", said
    assert outbox.address_span(said) is None, said
    assert outbox.unresolved_address(said) == heard, said


@pytest.mark.parametrize("said", [
    "heather at example dot com", "heather underscore smith at example dot com",
    "yes, send it to heather@example.com", "Heather", "to Heather at heather at gmail dot com",
    "<heather@example.com>", "mailto:heather@example.com", "no, to dana@example.com.",
])
def test_a_readable_address_or_a_name_is_not_unresolved(said):
    assert outbox.unresolved_address(said) == "", said


def test_prepare_says_what_it_heard_when_it_cannot_read_the_address(roots):
    cfg = cfg_with_roots(roots, **{"send_file.from": "school"})
    prep = outbox.prepare(cfg, None, "the biosensors handout",
                          "heather tilde smith at example dot com")
    assert prep.draft is None and prep.status == "No address"
    assert prep.ask == outbox.HEARD_LINE.format(
        heard="heather tilde smith at example dot com")
    assert "@" not in prep.ask and prep.ask.endswith("?")


# ---- (c) the 16 attack rows: read back AS SAID, or asked again ----------
_ATTACK_JOINER_ROWS = [
    ("yes, send it to heather underscore smith at example dot com",
     "heather_smith@example.com", "smith at"),
    ("yes, send it to heather dash smith at example dot com",
     "heather-smith@example.com", "smith at"),
    ("yes, send it to h underscore peyrovi at tamu dot edu",
     "h_peyrovi@tamu.edu", "peyrovi at"),
    ("yes, send it to heather hyphen smith at example dot com",
     "heather-smith@example.com", "smith at"),
    ("send it to heather underscore smith at example dot com instead",
     "heather_smith@example.com", "smith at"),
    ("no, to heather dash jones at example dot com",
     "heather-jones@example.com", "jones at"),
]


@pytest.mark.parametrize("source", ["typed", "voice"])
@pytest.mark.parametrize("said,to,cut", _ATTACK_JOINER_ROWS)
def test_a_joined_spoken_address_is_read_back_whole_and_the_yes_goes_there(
        cmd, said, to, cut, source):
    cmd.handle("email the biosensors handout to Heather", source=source)
    res = cmd.handle(said, source=source)
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.reply.endswith("Send it, sir?"), (said, res)
    assert outbox.spoken_address(to) in res.reply, (said, res.reply)
    assert f"to {cut}" not in res.reply, (said, res.reply)
    assert "@" not in res.reply
    cmd.handle("yes", source=source)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == to, said


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_the_read_back_of_an_underscored_address_said_back_sends_there(cmd, source):
    """Jarvis's own words: heather_smith@example.com is read back "to
    heather underscore smith at example dot com", and saying exactly that
    back is the read-back said back -- it sends THERE, not to smith@."""
    res = cmd.handle("email the biosensors handout to heather_smith@example.com",
                     source=source)
    assert res.reply.endswith("Send it, sir?")
    assert "heather underscore smith at example dot com" in res.reply, res.reply
    res = cmd.handle("yes, send it to heather underscore smith at example dot com",
                     source=source)
    assert res is not None and res.ack, res
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather_smith@example.com"


@pytest.mark.parametrize("source", ["typed", "voice"])
@pytest.mark.parametrize("said,heard", [
    ("yes, send it to heather tilde smith at example dot com",
     "heather tilde smith at example dot com"),
    ("yes, send it to heather plus lab at example dot com",
     "heather plus lab at example dot com"),
    ("no, to heather slash jones at example dot com",
     "heather slash jones at example dot com"),
])
def test_an_unreadable_spoken_address_is_asked_again_and_a_yes_then_sends_nothing(
        cmd, said, heard, source):
    cmd.handle("email the biosensors handout to Heather", source=source)
    res = cmd.handle(said, source=source)
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.speak, (said, res)
    assert res.reply == outbox.HEARD_LINE.format(heard=heard), (said, res.reply)
    assert "@" not in res.reply
    assert cmd._pending_send is None, "the draft to Heather is spent, not kept"
    assert cmd.question_open(), "the question left a slot behind it"
    res = cmd.handle("yes", source=source)
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.reply and "@" not in res.reply, res
    # the address said properly is read back to IT, and the yes goes there
    res = cmd.handle("heather underscore smith at example dot com", source=source)
    assert not FakeSMTP.made and res.reply.endswith("Send it, sir?"), res
    assert "heather underscore smith at example dot com" in res.reply
    cmd.handle("yes", source=source)
    assert FakeSMTP.made and FakeSMTP.made[-1].sent[0]["To"] == "heather_smith@example.com"


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_a_first_sentence_with_an_unreadable_address_is_asked_not_armed(cmd, source):
    res = cmd.handle("email the biosensors handout to heather tilde smith at example dot com",
                     source=source)
    assert not FakeSMTP.made
    assert res is not None and res.reply == outbox.HEARD_LINE.format(
        heard="heather tilde smith at example dot com"), res
    assert cmd._pending_send is None and cmd.question_open()
    res = cmd.handle("heather tilde smith at example dot com", source=source)
    assert not FakeSMTP.made and res.reply == outbox.HEARD_LINE.format(
        heard="heather tilde smith at example dot com"), res
    res = cmd.handle("yes", source=source)
    assert not FakeSMTP.made and res is not None and res.reply, res


# ---- (d) attack 2: the no-address line never speaks a raw "@" -----------
@pytest.mark.parametrize("source", ["typed", "voice"])
@pytest.mark.parametrize("said,who", [
    ("yes, send it to heather@example", "heather at example"),
    ("yes, to heather@", "heather at"),
])
def test_the_no_address_line_speaks_a_half_address(cmd, said, who, source):
    cmd.handle("email the biosensors handout to Heather", source=source)
    res = cmd.handle(said, source=source)
    assert not FakeSMTP.made, (said, res)
    assert res is not None and res.speak, (said, res)
    assert res.reply == outbox.NO_RECIPIENT_LINE.format(who=who), (said, res.reply)
    assert "@" not in res.reply
    assert cmd.question_open()
    cmd.handle("yes", source=source)
    assert not FakeSMTP.made


def test_spoken_who_leaves_a_name_alone_and_speaks_an_address():
    assert outbox.spoken_who("Heather Jones") == "Heather Jones"
    assert outbox.spoken_who("Mary-Jane") == "Mary-Jane"
    assert outbox.spoken_who("heather@example") == "heather at example"
    assert outbox.spoken_who("h_p@example.com") == "h underscore p at example dot com"


# ==================================================================
# 27. Spelled letter by letter, and the domain as ONE WORD (09-05)
# ==================================================================
# Hunter, 09-05: he spelled an address character by character, the
# capture was chopped into four turns and the tail "made weirdish
# nonsense". jarvis/spelling.py now holds the mic open across the letters
# and folds them into a word (tests/test_spelling_hold.py is the rule and
# the fold). THIS section drives what comes out of that through the send
# lane to the fake transport, because the fold and the parse are worth
# nothing unless the WIRE carries the address he spelled and nothing can
# go without the read-back.
#
# Whisper wrote his domain as ONE WORD -- "at example.com", never "at
# example dot com" -- so that is the shape every row here uses. HIS RULING
# (B), 09-05: it counts as a domain. Before the ruling (jarvis-v3, and this
# branch at 996408d) the one-word shape did not parse, unresolved_address
# handed nothing back either, and the send-intent gate
# (commander._send_names_someone) simply did not see an address: these
# tests are RED there and GREEN here.

# His fourth fragment in the exact shape whisper writes it: two spelled
# groups, whisper's comma after each, the domain run together, a full stop.
SPELLED_SAID = "q-z-v, k-b-w-7, at example.com."
SPELLED_ADDR = "qzvkbw7@example.com"
SPELLED_HEARD = outbox.spoken_address(SPELLED_ADDR)   # "qzvkbw7 at example dot com"


def _sent_to() -> list:
    """Every To: that reached the fake transport, in order."""
    return [m["To"] for conn in FakeSMTP.made for m in conn.sent]


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_a_spelled_address_with_a_one_word_domain_is_read_back_and_the_yes_goes_there(
        cmd, source):
    """The first sentence, whole: the letters fold, the one-word domain
    parses, the read-back speaks the address, and ONLY the yes sends --
    to exactly the address he spelled."""
    res = cmd.handle(f"email the biosensors handout to {SPELLED_SAID}", source=source)
    assert _sent_to() == [], "the read-back must not send"
    assert res is not None and res.handled and res.speak, res
    assert res.reply.endswith("Send it, sir?"), res.reply
    assert SPELLED_HEARD in res.reply, res.reply
    assert "@" not in res.reply
    assert cmd._pending_send is not None and cmd._pending_send.to_addr == SPELLED_ADDR
    cmd.handle("yes", source=source)
    assert _sent_to() == [SPELLED_ADDR]


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_a_spelled_answer_to_what_is_it_goes_on_the_wire_exactly(cmd, source):
    """"I've no address for Dana, sir. What is it?" -- and he spells it.
    The wire carries the address he spelled, character for character, and
    nothing before the yes."""
    res = cmd.handle("email the biosensors handout to Dana", source=source)
    assert res.reply == outbox.NO_RECIPIENT_LINE.format(who="Dana")
    res = cmd.handle(SPELLED_SAID, source=source)
    assert _sent_to() == []
    assert res is not None and res.reply.endswith("Send it, sir?"), res
    assert SPELLED_HEARD in res.reply and "@" not in res.reply, res.reply
    cmd.handle("yes", source=source)
    assert _sent_to() == [SPELLED_ADDR]


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_a_spelled_correction_of_a_read_back_is_read_back_again_before_anything_sends(
        cmd, source):
    """The send-intent gate itself (_send_names_someone): a correction
    that names a spelled address with a one-word domain is SEEN as an
    address now, so it is read back to the new address -- never sent to
    the old one, never dropped in silence."""
    cmd.handle("email the biosensors handout to Heather", source=source)
    res = cmd.handle(f"no, send it to {SPELLED_SAID}", source=source)
    assert _sent_to() == [], "a correction must not send to anyone"
    assert res is not None and res.speak and res.reply.endswith("Send it, sir?"), res
    assert SPELLED_HEARD in res.reply and "@" not in res.reply, res.reply
    assert "heather at example dot com" not in res.reply
    cmd.handle("yes", source=source)
    assert _sent_to() == [SPELLED_ADDR], "the yes goes to the corrected address only"


@pytest.mark.parametrize("source", ["typed", "voice"])
@pytest.mark.parametrize("said", [
    "z-v-k-b-w-7 at example",          # the top level lost
    "q-z-v at example dot",            # the domain cut off
    "at example.com",                  # the local part lost
    "q-z-v at example.c0m",            # a digit where the top level goes
    "q-z-v tilde k at example.com",    # a symbol word the parser does not read
])
def test_an_address_shape_that_will_not_parse_is_asked_about_out_loud_never_in_silence(
        cmd, said, source):
    """Silence is the defect. An answer with an address's SHAPE that the
    parser cannot make an address of is a spoken re-ask every time: the
    "I heard ..." line when the parser can hand back what it heard, the
    plain re-ask otherwise -- never None, never a raw "@", never a send."""
    cmd.handle("email the biosensors handout to Dana", source=source)
    res = cmd.handle(said, source=source)
    assert _sent_to() == [], (said, res)
    assert res is not None and res.handled and res.speak and res.reply, (said, res)
    assert "@" not in res.reply, (said, res.reply)
    assert res.reply.startswith("I heard ") or res.reply == outbox.ADDRESS_REASK_LINE, \
        (said, res.reply)
    assert cmd._pending_send is None and cmd.question_open(), (said, res)
    res = cmd.handle("yes", source=source)
    assert _sent_to() == [], "a yes to a re-ask sends nothing"
    assert res is not None and res.reply, (said, res)


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_a_first_sentence_whose_address_lost_its_top_level_says_what_it_heard(cmd, source):
    """Whisper drops the ".com": the sentence still names a real file, so
    it is his, and what he hears back names what was heard -- not silence,
    and not a draft to a half address."""
    res = cmd.handle("email the biosensors handout to z-v-k-b-w-7 at example",
                     source=source)
    assert _sent_to() == []
    assert res is not None and res.speak and res.reply, res
    assert res.reply == outbox.NO_RECIPIENT_LINE.format(who="z-v-k-b-w-7 at example"), res.reply
    assert cmd._pending_send is None and cmd.question_open()
    cmd.handle("yes", source=source)
    assert _sent_to() == []


# ---- the [1, 6] split: a KNOWN LIMIT, pinned rather than hidden ----------
# The hold reads the live preview's newest decode, and the preview runs at
# a 0.9 s cadence against a 0.8 s endpoint. When the decode of his first
# letter has not landed by the time the stop is due, that letter closes a
# capture of its own and the other six open the next one. The second
# capture then drafts a PLAUSIBLE address missing its first letter --
# where the baseline's four-way chop produced obvious nonsense. This is a
# new failure shape; jarvis/spelling.py cannot see across captures and does
# not try to. The ONLY thing that catches it is the spoken read-back.
SPLIT_FIRST = "email the biosensors handout to q"        # capture 1: the first letter alone
SPLIT_REST = "z-v-k-b-w-7 at example.com"                # capture 2: the other six
SPLIT_WRONG = "zvkbw7@example.com"                       # plausible, and missing its q


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_the_one_six_split_drafts_a_plausible_wrong_address_and_he_hears_it_first(
        cmd, source):
    """The failure shape, driven end to end. The wrong address is READ
    BACK before anything can happen to it, and a no sends nothing."""
    res = cmd.handle(SPLIT_FIRST, source=source)
    assert res is not None and res.speak and res.reply, res
    assert cmd._pending_send is None and cmd.question_open(), "a bare letter arms no draft"
    res = cmd.handle(SPLIT_REST, source=source)
    assert _sent_to() == []
    assert res is not None and res.reply.endswith("Send it, sir?"), res
    assert outbox.spoken_address(SPLIT_WRONG) in res.reply, res.reply   # he HEARS the missing q
    assert "@" not in res.reply
    assert cmd._pending_send is not None and cmd._pending_send.to_addr == SPLIT_WRONG
    res = cmd.handle("no", source=source)
    assert res.reply == outbox.DROPPED_LINE
    assert _sent_to() == [] and cmd._pending_send is None


@pytest.mark.parametrize("source", ["typed", "voice"])
def test_nothing_reassembled_goes_on_the_wire_unread(cmd, source):
    """Both counts, pinned: zero sends before the read-back, one after the
    yes, and the To: on the wire is the SAME address the read-back spoke
    -- so a wrong address can only ever go out after he has heard it."""
    cmd.handle(SPLIT_FIRST, source=source)
    res = cmd.handle(SPLIT_REST, source=source)
    heard = res.reply
    assert _sent_to() == []
    cmd.handle("yes", source=source)
    assert _sent_to() == [SPLIT_WRONG]
    assert outbox.spoken_address(_sent_to()[0]) in heard, (heard, _sent_to())


def test_every_draft_is_read_back_before_it_can_be_sent_by_construction():
    """The read-back is unconditional by CONSTRUCTION, not by case: every
    draft in the send lane is armed by one function, _send_file_finish,
    which returns outbox.read_back for it, and outbox.send has exactly one
    caller in the commander, behind the yes to a pending draft. A second
    arming point or a second send call is a hole in the [1, 6] catch, and
    this test is what makes adding one deliberate."""
    import inspect
    import jarvis.commander as commander_mod
    src = inspect.getsource(commander_mod)
    assert len(re.findall(r"\boutbox\.send\(", src)) == 1, "one call to outbox.send"
    arms = [ln for ln in src.splitlines() if re.search(r"\.stash_send\(", ln)]
    assert len(arms) == 1, arms
    finish = inspect.getsource(commander_mod._send_file_finish)
    assert ".stash_send(" in finish and "outbox.read_back(prep.draft)" in finish
