"""THE NO-SPOKEN-"@" INVARIANT, pinned on every path a recipient is said.

Jarvis must never say an at sign out loud. It is not decoration: neither
engine is reliable on a raw address -- edge-tts spells some domains letter
by letter, and F5 clones prosody from a reference clip that has never said
an "@" -- so an address he HEARS has to be an address in words.
outbox.spoken_address is the function that does it, and outbox.spoken_who
the one that decides whether a recipient needs it (its docstring records
the round-4 attack where the no-address line echoed a typed "@").

WHAT BROKE IT. The sent-log carried onto this branch built the day's
summary as ``row.get("to_name") or str(row.get("to") or "")`` -- the
masked address, verbatim, "@" and all. The source branch (7ae1448) had
the spoken_address call and the port dropped it, on the reasoning that
the log is masked on disk. That conflates two different jobs: masking
stops a LEAK, spoken_address stops a SPOKEN at sign. Measured before the
fix, "what did I email today" answered:

    One, sir: lab report.pdf to d…@example.com.

WHY THESE ROWS ARE WHERE THEY ARE. The one sentence that was caught is
not the invariant; the invariant is that no line Jarvis speaks carries an
"@" from a recipient, whichever of them said it. So the fix is one seam --
outbox.spoken_recipient -- and these rows walk every path a recipient
reaches speech on: the draft read-back, a correction, the re-ask, the
day's audit summary, the rehearsal, and the auth failure. A future call
site that builds ``to_name or spoken_address(...)`` by hand fails
test_no_call_site_builds_a_spoken_recipient_by_hand.
"""
import json
import smtplib
import time
import types
from unittest.mock import MagicMock

import pytest

from jarvis import outbox
from jarvis.commander import Commander, IntentClassifier
from jarvis.config import CONFIG


# ============================================================
# harness -- the same shape tests/test_sent_today_and_subject.py uses
# ============================================================

@pytest.fixture
def home(tmp_path):
    import os
    for name in ("Desktop", "Downloads", "Documents"):
        (tmp_path / name).mkdir()
    now = time.time()
    for i, (rel, data) in enumerate({
        "Desktop/lab_report.pdf": b"L" * 3000,
        "Documents/Biosensors_Lab-Handout_v2.pdf": b"B" * 5000,
    }.items()):
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
    made: list = []

    def __init__(self, host, port, timeout=None):
        self.sent = []
        FakeSMTP.made.append(self)

    def login(self, user, password):
        pass

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        pass


class _AuthRefusingSMTP:
    def __init__(self, host, port, timeout=None):
        pass

    def login(self, user, password):
        raise smtplib.SMTPAuthenticationError(
            535, b"5.7.8 Username and Password not accepted for hp@tamu.edu")

    def send_message(self, msg):                     # pragma: no cover
        raise AssertionError("must not be reached")

    def quit(self):
        pass


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    """No test may append to his real memory/sent.jsonl."""
    FakeSMTP.made = []
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
    spoken: list = []
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


def _draft(tmp_path, to_addr="hjones@example.com", to_name="Heather",
           name="lab_report.pdf", label="school"):
    root = tmp_path / "Desktop"
    root.mkdir(exist_ok=True)
    f = root / name
    f.write_bytes(b"%PDF-1.4 body")
    st = f.stat()
    return outbox.Draft(
        path=f, size=st.st_size, mtime=st.st_mtime,
        to_addr=to_addr, to_name=to_name,
        account={"label": label, "address": "hp@tamu.edu",
                 "password": "school-secret"},
        subject="Lab report", roots=[str(root)], made_at=time.monotonic())


# ============================================================
# 0. THE SEAM ITSELF
# ============================================================

def test_the_seam_leaves_a_name_alone():
    assert outbox.spoken_recipient("Heather Jones", "hjones@example.com") \
        == "Heather Jones"
    assert outbox.spoken_recipient("Mary-Jane", "mj@example.com") == "Mary-Jane"


def test_the_seam_speaks_an_address_that_arrived_in_the_NAME_slot():
    """The round-4 attack: a half address he typed becomes the draft's
    ``to_name``, and every line that says ``to_name or ...`` then speaks
    its "@"."""
    assert outbox.spoken_recipient("hjones@example", "hjones@example.com") \
        == "hjones at example"
    assert "@" not in outbox.spoken_recipient("heather@", "h@example.com")


def test_the_seam_speaks_the_address_when_there_is_no_name():
    assert outbox.spoken_recipient("", "hjones@example.com") \
        == "hjones at example dot com"
    assert outbox.spoken_recipient(None, "hjones@example.com") \
        == "hjones at example dot com"


def test_the_seam_speaks_a_MASKED_address_without_its_at():
    """What the audit holds is masked -- "h…@example.com" -- and masking
    is not speaking: the "@" is still there to be said."""
    said = outbox.spoken_recipient("", "h…@example.com")
    assert "@" not in said and " at " in said


def test_the_seam_says_nothing_for_nothing():
    assert outbox.spoken_recipient("", "") == ""


# ============================================================
# 1. THE DAY'S AUDIT SUMMARY -- the line that was caught
# ============================================================

def test_the_days_summary_does_not_speak_an_at_sign(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, to_name=""), path=log)
    said = outbox.sent_today_line(path=log)
    assert "@" not in said, said
    assert " at " in said, said


def test_the_days_summary_still_prefers_the_name(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, to_name="Heather"), path=log)
    said = outbox.sent_today_line(path=log)
    assert "Heather" in said and "@" not in said


def test_a_half_address_in_the_name_column_is_spoken_not_said(tmp_path):
    """``to_name`` is masked on the way in, but mask_addresses only masks
    a WHOLE address: "heather@" survives it and reaches speech raw."""
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, to_name="heather@"), path=log)
    assert "@" not in outbox.sent_today_line(path=log)


def test_several_sends_in_one_sentence_speak_no_at_sign(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, to_name="", name="one.pdf"), path=log)
    outbox.record_sent(_draft(tmp_path, to_name="", name="two.pdf",
                              to_addr="dana@example.com"), path=log)
    said = outbox.sent_today_line(path=log)
    assert "@" not in said and said.startswith("2, sir:")


def test_end_to_end_the_audit_answer_speaks_no_at_sign(cmd):
    """Through Commander.handle, the ordinary path: an address he typed,
    a yes, then the question."""
    cmd.handle("email the lab report to dana@example.com", source="typed")
    assert cmd._pending_send is not None
    cmd.handle("yes", source="typed")
    assert FakeSMTP.made, "the send must actually have happened"
    res = cmd.handle("what did I email today", source="typed")
    assert res.handled and res.speak
    assert "@" not in res.reply, res.reply
    assert "lab report" in res.reply


def test_end_to_end_an_address_he_SAID_is_read_back_in_words(cmd):
    """The shape the verdict measured: he says the address aloud, the
    parser drafts it, the mask writes "d…", and the summary says it."""
    cmd.handle("email the lab report to dana at example dot com",
               source="typed")
    assert cmd._pending_send is not None
    cmd.handle("yes", source="typed")
    res = cmd.handle("what did I email today", source="typed")
    assert "@" not in res.reply, res.reply


# ============================================================
# 2. THE DRAFT READ-BACK
# ============================================================

def test_the_read_back_of_a_typed_address_speaks_no_at_sign(tmp_path):
    said = outbox.read_back(_draft(tmp_path, to_name=""))
    assert "@" not in said and "hjones at example dot com" in said


def test_the_read_back_speaks_a_half_address_that_became_the_name(tmp_path):
    said = outbox.read_back(_draft(tmp_path, to_name="hjones@example"))
    assert "@" not in said, said


def test_end_to_end_the_read_back_speaks_no_at_sign(cmd):
    res = cmd.handle("email the lab report to dana@example.com", source="typed")
    assert res.reply.endswith("Send it, sir?")
    assert "@" not in res.reply, res.reply


def test_the_account_half_of_the_read_back_speaks_no_at_sign(tmp_path):
    """account_label returns the configured label RAW; a label that is an
    address would be spoken in "from your ... account"."""
    said = outbox.account_words({"label": "hp@tamu.edu",
                                 "address": "hp@tamu.edu"})
    assert "@" not in said, said


# ============================================================
# 3. THE CORRECTION
# ============================================================

def test_end_to_end_a_correction_is_read_back_without_an_at_sign(cmd):
    cmd.handle("email the lab report to heather", source="typed")
    res = cmd.handle("no, send it to dana@example.com", source="typed")
    assert res is not None and res.handled
    assert not FakeSMTP.made
    assert "@" not in res.reply, res.reply


def test_the_redirect_to_himself_speaks_no_at_sign(cmd):
    """SELF_LINE: "send it to me" is never a yes, and he hears whom the
    draft is to."""
    cmd.handle("email the lab report to dana@example.com", source="typed")
    res = cmd.handle("send it to me", source="typed")
    assert not FakeSMTP.made and cmd._pending_send is not None
    assert "not to you" in res.reply
    assert "@" not in res.reply, res.reply


def test_the_redirect_speaks_a_half_address_in_the_name_slot(cmd, tmp_path):
    draft = _draft(tmp_path, to_name="dana@example")
    cmd._pending_send = draft
    res = cmd.handle("send it to me", source="typed")
    assert res is not None and res.handled
    assert "@" not in res.reply, res.reply


# ============================================================
# 4. THE RE-ASK
# ============================================================

def test_the_unsure_re_ask_speaks_no_at_sign(tmp_path):
    said = outbox.unsure_line(_draft(tmp_path, to_name=""))
    assert "@" not in said and "hjones at example dot com" in said


def test_the_unsure_re_ask_speaks_a_half_address_name(tmp_path):
    assert "@" not in outbox.unsure_line(_draft(tmp_path,
                                                to_name="hjones@example"))


def test_the_gender_re_ask_speaks_no_at_sign(tmp_path):
    assert "@" not in outbox.gender_line(_draft(tmp_path, to_name=""))


def test_the_gender_re_ask_speaks_a_half_address_name(tmp_path):
    assert "@" not in outbox.gender_line(_draft(tmp_path,
                                                to_name="hjones@example"))


def test_end_to_end_a_vague_answer_is_re_asked_without_an_at_sign(cmd):
    cmd.handle("email the lab report to dana@example.com", source="typed")
    res = cmd.handle("okay", source="typed")
    assert not FakeSMTP.made
    assert res.status == "Confirm?"
    assert "@" not in res.reply, res.reply


def test_the_no_address_re_ask_speaks_no_at_sign():
    """Already guarded by spoken_who -- pinned so it stays guarded."""
    line = outbox.NO_RECIPIENT_LINE.format(who=outbox.spoken_who("heather@"))
    assert "@" not in line


# ============================================================
# 5. THE REHEARSAL
# ============================================================

def test_the_rehearsal_line_speaks_no_at_sign(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_MAIL_DRYRUN", "1")
    line = outbox.send(_draft(tmp_path, to_name=""), smtp=FakeSMTP)
    assert line == outbox.REHEARSAL_LINE
    assert "@" not in line
    assert FakeSMTP.made == []


def test_a_rehearsal_leaves_nothing_for_the_summary_to_say(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("JARVIS_MAIL_DRYRUN", "1")
    outbox.send(_draft(tmp_path, to_name=""), smtp=FakeSMTP)
    said = outbox.sent_today_line(path=outbox.SENT_LOG)
    assert said == "Nothing today, sir." and "@" not in said


def test_the_sent_line_after_a_real_send_speaks_no_at_sign(tmp_path):
    line = outbox.send(_draft(tmp_path, to_name=""), smtp=FakeSMTP)
    assert line != outbox.REHEARSAL_LINE
    assert "@" not in line, line


def test_the_sent_line_speaks_a_half_address_name(tmp_path):
    line = outbox.send(_draft(tmp_path, to_name="hjones@example"),
                       smtp=FakeSMTP)
    assert "@" not in line, line


# ============================================================
# 6. THE AUTH FAILURE
# ============================================================

def test_the_auth_failure_line_speaks_no_at_sign():
    assert "@" not in outbox.auth_failed_line("school")
    assert "password" in outbox.auth_failed_line("school")


def test_the_auth_failure_line_speaks_an_address_shaped_label():
    """account_label falls back to the configured label RAW, so a label
    he set to his address would be read out with its "@"."""
    said = outbox.auth_failed_line("hp@tamu.edu")
    assert "@" not in said, said


def test_the_auth_failure_line_has_a_word_when_there_is_no_label():
    assert "@" not in outbox.auth_failed_line("")
    assert outbox.auth_failed_line("").strip()


def test_end_to_end_a_refused_password_speaks_no_at_sign(cmd):
    cmd.services.smtp = _AuthRefusingSMTP
    cmd.handle("email the lab report to dana@example.com", source="typed")
    assert cmd._pending_send is not None
    cmd.handle("yes", source="typed")
    said = " ".join(cmd.spoken)
    assert "password" in said.lower(), said
    assert "@" not in said, said


def test_the_wire_failure_line_speaks_no_at_sign():
    assert "@" not in outbox.WIRE_FAILED_LINE


# ============================================================
# 7. THE SEAM IS THE ONLY WAY -- no call site rebuilds it by hand
# ============================================================

def test_no_call_site_builds_a_spoken_recipient_by_hand():
    """``to_name or spoken_address(addr)`` is the shape that lost the
    invariant: it speaks ``to_name`` untouched. There is one seam now,
    and this row fails the moment a new call site grows its own."""
    import ast
    import pathlib
    root = pathlib.Path(outbox.__file__).resolve().parent
    offenders = []

    def _is_spoken_address(node):
        fn = node.func if isinstance(node, ast.Call) else None
        if isinstance(fn, ast.Name):
            return fn.id == "spoken_address"
        return isinstance(fn, ast.Attribute) and fn.attr == "spoken_address"

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                            # pragma: no cover
            continue
        for node in ast.walk(tree):
            # ``<anything>.to_name or spoken_address(...)`` -- the shape,
            # read off the syntax tree so a comment about it is not a hit.
            if not (isinstance(node, ast.BoolOp)
                    and isinstance(node.op, ast.Or) and len(node.values) == 2):
                continue
            left, right = node.values
            if (isinstance(left, ast.Attribute) and left.attr == "to_name"
                    and _is_spoken_address(right)):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], offenders


def test_every_spoken_line_constant_is_free_of_an_at_sign():
    """The templates themselves. A "{who}" filled by the seam is safe;
    an "@" baked into the sentence never could be."""
    offenders = [name for name in dir(outbox)
                 if name.isupper() and name.endswith(("_LINE", "_STATUS"))
                 and isinstance(getattr(outbox, name), str)
                 and "@" in getattr(outbox, name)]
    assert offenders == [], offenders


def test_nothing_in_the_audit_summary_reads_an_unmasked_field(tmp_path):
    """Belt and braces on the OTHER invariant: the row on disk is masked
    on the way in, and the spoken line is built from that row."""
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, to_addr="hjones@example.com",
                              to_name=""), path=log)
    raw = log.read_text(encoding="utf-8")
    assert "hjones@example.com" not in raw
    row = json.loads(raw.splitlines()[0])
    assert row["to"].startswith("h") and "hjones" not in row["to"]
    assert "hjones@example.com" not in outbox.sent_today_line(path=log)
