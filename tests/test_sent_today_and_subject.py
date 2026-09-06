"""Two commander gaps carried from the email-a-file branch (7ae1448).

1.  "What did I email today?" had no answer at all. The audit log
    (outbox.SENT_LOG) is what replaces an unsend, and an audit nobody can
    ask about is a file, not a feature.

2.  A SPOKEN SUBJECT could never reach a draft. outbox.prepare() has
    accepted ``subject=`` all along, but ``git grep 'subject=' --
    jarvis/commander.py`` returned ZERO call sites on jarvis-v3, so
    "email the lab report to Heather with the subject week nine" put the
    subject nowhere and sent the default. Measured, not assumed.
"""
import time
import types
from unittest.mock import MagicMock

import pytest

from jarvis import outbox
from jarvis.commander import (Commander, IntentClassifier, REGISTRY,
                              _SEND_FILE_RX)
from jarvis.config import CONFIG


@pytest.fixture
def home(tmp_path):
    for name in ("Desktop", "Downloads", "Documents"):
        (tmp_path / name).mkdir()
    files = {
        "Desktop/lab_report.pdf": b"L" * 3000,
        "Documents/Biosensors_Lab-Handout_v2.pdf": b"B" * 5000,
    }
    import os
    now = time.time()
    for i, (rel, data) in enumerate(files.items()):
        p = tmp_path / rel
        p.write_bytes(data)
        os.utime(p, (now - i * 3600, now - i * 3600))
    return tmp_path


@pytest.fixture
def roots(home):
    return [str(home / "Desktop"), str(home / "Downloads"),
            str(home / "Documents")]


class Cfg:
    def __init__(self, **over):
        self.data = {
            "gmail.accounts": [
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
    made = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.logged_in = None
        self.sent = []
        FakeSMTP.made.append(self)

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        pass


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    FakeSMTP.made = []
    # The audit must never touch his real memory dir from a test.
    monkeypatch.setattr(outbox, "SENT_LOG", tmp_path / "sent.jsonl")
    yield
    FakeSMTP.made = []


@pytest.fixture
def cmd(tmp_path, monkeypatch, roots):
    monkeypatch.setattr(IntentClassifier, "INTENT_LOG", tmp_path / "intent.json")
    monkeypatch.setattr(Commander, "FEEDBACK_LOG", tmp_path / "fb.jsonl",
                        raising=False)
    for key, val in (("voice_cmds", True), ("jarvis_mode", True),
                     ("auto_type", False), ("talkback", False)):
        monkeypatch.setattr(CONFIG, key, val)
    monkeypatch.setattr(Commander, "_bg", lambda self, fn: fn())
    spoken = []
    monkeypatch.setattr(Commander, "_speak_now",
                        lambda self, text: spoken.append(text) or True)
    svc = types.SimpleNamespace(
        assistant=Cfg(**{"send_file.roots": roots, "send_file.from": "school"}),
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


# ============================================================
# 1. "What did I email today?"
# ============================================================

def test_the_sent_files_command_is_registered():
    assert "sent files" in [x.name for x in REGISTRY]


@pytest.mark.parametrize("said", [
    "what did I email today",
    "what did I send today",
    "what files did I email today",
    "what have I emailed today",
    "what did you email today",
])
def test_the_question_is_heard_in_its_usual_shapes(cmd, said):
    res = cmd.handle(said, source="typed")
    assert res is not None and res.handled


def test_with_nothing_sent_it_says_so(cmd):
    res = cmd.handle("what did I email today", source="typed")
    assert res.reply == "Nothing today, sir."


def test_it_reads_the_audit_back(cmd, tmp_path):
    d = outbox.Draft(path=tmp_path / "lab_report.pdf", size=10, mtime=0,
                     to_addr="heather@example.com", to_name="Heather",
                     account={"label": "school", "address": "hp@tamu.edu"},
                     subject="Lab report")
    outbox.record_sent(d, path=outbox.SENT_LOG)
    res = cmd.handle("what did I email today", source="typed")
    assert "lab report" in res.reply and "Heather" in res.reply


def test_the_answer_never_speaks_a_raw_address(cmd, tmp_path):
    d = outbox.Draft(path=tmp_path / "lab_report.pdf", size=10, mtime=0,
                     to_addr="heather@example.com", to_name="",
                     account={"label": "school", "address": "hp@tamu.edu"},
                     subject="Lab report")
    outbox.record_sent(d, path=outbox.SENT_LOG)
    res = cmd.handle("what did I email today", source="typed")
    assert "heather@example.com" not in res.reply


def test_the_question_does_not_arm_a_send(cmd):
    """It is a QUESTION about sending. It must not be caught by the send
    family and turned into a draft."""
    cmd.handle("what did I email today", source="typed")
    assert getattr(cmd, "_pending_send", None) is None


# ============================================================
# 2. A spoken subject reaches the draft
# ============================================================

@pytest.mark.parametrize("said,want", [
    ("email the lab report to heather with the subject week nine",
     "week nine"),
    ("email the lab report to heather with subject week nine", "week nine"),
    ("email the lab report to heather, subject line week nine", "week nine"),
    ("email the lab report to heather subject: week nine", "week nine"),
])
def test_the_regex_captures_a_spoken_subject(said, want):
    m = _SEND_FILE_RX.match(said)
    assert m is not None
    assert (m.group("subj") or m.group("subj2") or "").strip() == want


def test_the_subject_does_not_swallow_the_recipient():
    m = _SEND_FILE_RX.match(
        "email the lab report to heather with the subject week nine")
    assert m.group("who_a").strip() == "heather"
    assert m.group("file_a").strip() == "the lab report"


def test_a_send_with_no_subject_still_parses():
    m = _SEND_FILE_RX.match("email the lab report to heather")
    assert m is not None
    assert not (m.group("subj") or m.group("subj2"))


def test_the_subject_survives_an_account_hint():
    """The cue is accepted on EITHER side of "from my school account",
    because he says it both ways."""
    for said in (
        "email the lab report to heather with the subject week nine "
        "from my school account",
        "email the lab report to heather from my school account "
        "with the subject week nine",
    ):
        m = _SEND_FILE_RX.match(said)
        assert m is not None, said
        assert (m.group("subj") or m.group("subj2") or "").strip() \
            == "week nine", said
        assert m.group("acct").strip() == "school", said


def test_the_spoken_subject_reaches_the_draft(cmd):
    res = cmd.handle("email the lab report to heather with the subject "
                     "week nine", source="typed")
    assert res.handled
    draft = cmd._pending_send
    assert draft is not None
    assert draft.subject == "week nine"


def test_without_a_subject_the_default_is_still_used(cmd):
    res = cmd.handle("email the lab report to heather", source="typed")
    assert res.handled
    assert cmd._pending_send.subject
    assert cmd._pending_send.subject != "week nine"


# ============================================================
# 3. A refused password does not sound like a dead network
# ============================================================

class _AuthRefusingSMTP:
    def __init__(self, host, port, timeout=None):
        pass

    def login(self, user, password):
        import smtplib
        raise smtplib.SMTPAuthenticationError(
            535, b"5.7.8 Username and Password not accepted")

    def send_message(self, msg):                     # pragma: no cover
        raise AssertionError("must not be reached")

    def quit(self):
        pass


class _DeadWireSMTP:
    def __init__(self, host, port, timeout=None):
        raise OSError("network is unreachable")


def _arm(cmd):
    res = cmd.handle("email the lab report to heather", source="typed")
    assert cmd._pending_send is not None, res
    return res


def test_a_refused_password_names_the_password(cmd):
    cmd.services.smtp = _AuthRefusingSMTP
    _arm(cmd)
    cmd.handle("yes", source="typed")
    said = " ".join(cmd.spoken)
    assert "password" in said.lower()
    assert "school" in said.lower()


def test_a_dead_wire_does_NOT_blame_the_password(cmd):
    cmd.services.smtp = _DeadWireSMTP
    _arm(cmd)
    cmd.handle("yes", source="typed")
    said = " ".join(cmd.spoken)
    assert "password" not in said.lower()


def test_both_failures_promise_nothing_was_sent(cmd):
    for transport in (_AuthRefusingSMTP, _DeadWireSMTP):
        cmd.services.smtp = transport
        cmd.spoken.clear()
        _arm(cmd)
        cmd.handle("yes", source="typed")
        said = " ".join(cmd.spoken).lower()
        assert "nothing" in said, (transport, said)


def test_a_failed_send_is_not_in_the_audit(cmd):
    cmd.services.smtp = _AuthRefusingSMTP
    _arm(cmd)
    cmd.handle("yes", source="typed")
    assert outbox.sent_rows(outbox.SENT_LOG) == []
    assert cmd.handle("what did I email today",
                      source="typed").reply == "Nothing today, sir."


def test_a_good_send_IS_in_the_audit_and_can_be_asked_about(cmd):
    _arm(cmd)
    cmd.handle("yes", source="typed")
    res = cmd.handle("what did I email today", source="typed")
    # The recipient is named as the CONTACTS MAP spells it, not as the
    # book would capitalise it.
    assert "lab report" in res.reply and "heather" in res.reply.lower()
    assert "One, sir" in res.reply
