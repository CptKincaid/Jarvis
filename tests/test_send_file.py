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
    with pytest.raises(ConnectionRefusedError) as exc:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            ("smtp.gmail.com", 465))
    assert "SMTP" in str(exc.value)


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
    svc = types.SimpleNamespace(
        assistant=cfg_with_roots(roots, **{"send_file.from": "school"}),
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
    assert res.reply == outbox.UNSURE_LINE and not FakeSMTP.made
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
