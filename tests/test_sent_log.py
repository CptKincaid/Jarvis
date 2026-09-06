"""The audit of what actually left the machine.

Carried by hand from the email-a-file branch (7ae1448) onto today's
outbox. THIS IS WHAT REPLACES AN UNDO: there is no unsend over SMTP --
Gmail's is a delay its own web client implements and its submission
server knows nothing about -- so a record of what left, when and to whom
is the only honest substitute.

The one thing the source branch got WRONG and this port fixes: 7ae1448's
record_sent wrote ``"to": str(draft.to_addr)`` RAW. The address-book lane
rewrote outbox.mask_addresses tonight precisely so that an address the
parser can draft never reaches a log line raw, and a sent log that wrote
addresses raw would reopen exactly that hole in a NEW file. Every address
this log writes goes through mask_addresses on the way IN, so the raw
address is never on disk at all -- masking on the way out would still
leave it in the file.
"""
import json
import time
from datetime import datetime

import pytest

from jarvis import outbox


def _draft(tmp_path, to_addr="hjones@example.com", to_name="Heather",
           subject="Lab report", name="lab_report.pdf"):
    f = tmp_path / name
    f.write_bytes(b"%PDF-1.4 body")
    return outbox.Draft(
        path=f, size=f.stat().st_size, mtime=f.stat().st_mtime,
        to_addr=to_addr, to_name=to_name,
        account={"label": "school", "address": "hp@tamu.edu",
                 "password": "school-secret"},
        subject=subject, made_at=time.monotonic())


# ============================================================
# 1. It records what left
# ============================================================

def test_record_sent_appends_one_json_line(tmp_path):
    log = tmp_path / "memory" / "sent.jsonl"
    assert outbox.record_sent(_draft(tmp_path), message_id="<abc@x>",
                              path=log) is True
    rows = [json.loads(ln) for ln in
            log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "lab_report.pdf"
    assert row["account"] == "school"
    assert row["message_id"] == "<abc@x>"
    assert row["dry_run"] is False
    assert row["bytes"] == 13


def test_the_log_holds_no_body_and_no_password(tmp_path):
    log = tmp_path / "sent.jsonl"
    d = _draft(tmp_path)
    d.body = "the confidential body text"
    outbox.record_sent(d, path=log)
    text = log.read_text(encoding="utf-8")
    assert "confidential" not in text
    assert "school-secret" not in text


def test_two_sends_append_rather_than_overwrite(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, name="one.pdf"), path=log)
    outbox.record_sent(_draft(tmp_path, name="two.pdf"), path=log)
    assert len(outbox.sent_rows(log)) == 2


def test_a_log_that_cannot_be_written_is_not_a_failed_send(tmp_path):
    """An audit that fails must never be the reason a send is reported as
    failed: the message HAS gone, and saying otherwise is the exact lie
    this lane exists to prevent."""
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")
    assert outbox.record_sent(_draft(tmp_path),
                              path=blocked / "sent.jsonl") is False


def test_the_log_is_written_owner_only(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path), path=log)
    assert oct(log.stat().st_mode)[-3:] == "600"


# ============================================================
# 2. Every address goes in MASKED -- the hole 7ae1448 left open
# ============================================================

def test_the_recipient_address_is_masked_on_the_way_in(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, to_addr="hjones@example.com"),
                       path=log)
    text = log.read_text(encoding="utf-8")
    assert "hjones@example.com" not in text
    assert "h…@example.com" in text


def test_a_SPOKEN_address_in_the_name_slot_is_masked_too(tmp_path):
    """When he says the address there is no validated name, so the
    spoken form lands in to_name -- and the parser can draft from it,
    which is exactly what mask_addresses keys on."""
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(
        _draft(tmp_path, to_addr="dana@example.com",
               to_name="dana at example dot com"), path=log)
    text = log.read_text(encoding="utf-8")
    assert "dana at example dot com" not in text
    assert "d… at example dot com" in text


def test_an_address_hidden_in_the_subject_is_masked(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, subject="fwd to dana@example.com"),
                       path=log)
    assert "dana@example.com" not in log.read_text(encoding="utf-8")


def test_the_mask_is_exactly_the_modules_own(tmp_path):
    """Pinned to mask_addresses itself, not to a copy of its output, so
    anything the parser learns to read tomorrow is masked here in the
    same edit."""
    log = tmp_path / "sent.jsonl"
    raw = "hjones@example.com"
    outbox.record_sent(_draft(tmp_path, to_addr=raw, to_name=""), path=log)
    assert outbox.sent_rows(log)[0]["to"] == outbox.mask_addresses(raw)


# ============================================================
# 3. Reading it back
# ============================================================

def test_sent_rows_survives_a_corrupt_line(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path), path=log)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("{not json at all\n\n")
    outbox.record_sent(_draft(tmp_path, name="two.pdf"), path=log)
    assert len(outbox.sent_rows(log)) == 2


def test_sent_rows_on_a_missing_file_is_empty(tmp_path):
    assert outbox.sent_rows(tmp_path / "nope.jsonl") == []


def test_sent_today_line_with_nothing_sent(tmp_path):
    assert outbox.sent_today_line(path=tmp_path / "nope.jsonl") \
        == "Nothing today, sir."


def test_sent_today_line_names_the_one_file_and_the_person(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path), path=log)
    said = outbox.sent_today_line(path=log)
    assert "One, sir" in said and "lab report" in said and "Heather" in said


def test_sent_today_line_counts_and_joins_several(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, name="one.pdf"), path=log)
    outbox.record_sent(_draft(tmp_path, name="two.pdf", to_name="Dana"),
                       path=log)
    said = outbox.sent_today_line(path=log)
    assert said.startswith("2, sir:") and " and " in said


def test_yesterdays_send_is_not_todays(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path), path=log)
    rows = log.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["at"] = "2001-01-01T09:00:00"
    log.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert outbox.sent_today_line(path=log) == "Nothing today, sir."


def test_a_REHEARSAL_is_never_listed_as_something_he_sent(tmp_path):
    """Nothing left the machine, so listing one among the day's sends
    would be the audit telling the same lie the spoken line avoids."""
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path), dry_run=True, path=log)
    assert outbox.sent_today_line(path=log) == "Nothing today, sir."
    assert outbox.sent_rows(log)[0]["dry_run"] is True


def test_the_spoken_summary_never_says_a_raw_address(tmp_path):
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, to_addr="hjones@example.com",
                              to_name=""), path=log)
    assert "hjones@example.com" not in outbox.sent_today_line(path=log)


# ============================================================
# 4. The read-back helpers (spoken_when / subfolder_words)
# ============================================================

def test_spoken_when_reads_the_clock_in_words(tmp_path):
    now = datetime(2026, 9, 5, 15, 0).timestamp()
    assert outbox.spoken_when(now - 30 * 60, now) == "saved in the last hour"
    assert outbox.spoken_when(
        datetime(2026, 9, 5, 9, 0).timestamp(), now) == "saved this morning"
    assert outbox.spoken_when(
        datetime(2026, 9, 4, 9, 0).timestamp(), now) == "saved yesterday"
    # 6 days back is still inside the weekday band...
    assert outbox.spoken_when(
        datetime(2026, 8, 30, 9, 0).timestamp(), now) == "saved on Sunday"
    # ...and past a week it becomes the date, said in words.
    assert outbox.spoken_when(
        datetime(2026, 8, 20, 9, 0).timestamp(), now) == "saved on 20 August"


def test_spoken_when_never_says_a_timestamp(tmp_path):
    said = outbox.spoken_when(datetime(2026, 8, 30, 18, 42).timestamp(),
                              datetime(2026, 9, 5, 15, 0).timestamp())
    assert "2026" not in said and ":" not in said


def test_spoken_when_on_rubbish_is_empty():
    assert outbox.spoken_when("not a time") == ""
    assert outbox.spoken_when(None) == ""


def test_subfolder_words_names_a_real_subfolder(tmp_path):
    root = tmp_path / "Desktop"
    sub = root / "Fall2026"
    sub.mkdir(parents=True)
    f = sub / "lab_report.pdf"
    f.write_bytes(b"x")
    d = _draft(tmp_path)
    d.path, d.roots = f, [root]
    assert outbox.subfolder_words(d) == "Fall2026"


def test_subfolder_words_is_empty_when_the_file_sits_in_a_root(tmp_path):
    root = tmp_path / "Desktop"
    root.mkdir()
    f = root / "lab_report.pdf"
    f.write_bytes(b"x")
    d = _draft(tmp_path)
    d.path, d.roots = f, [root]
    assert outbox.subfolder_words(d) == ""


# ============================================================
# 5. The spoken subject
# ============================================================

@pytest.mark.parametrize("said,want", [
    ("lab report", "lab report"),
    ("  spare   spaces  ", "spare spaces"),
    ("the lab report.", "the lab report"),
    ("", ""),
    (None, ""),
])
def test_clean_subject_trims_but_does_not_reword(said, want):
    """It does NOT strip the spoken cue -- commander's _SUBJECT_CUE has
    already consumed that before the text arrives here."""
    assert outbox.clean_subject(said) == want


def test_clean_subject_is_capped():
    """It is READ BACK VERBATIM, and a subject long enough to lose him is
    a subject he stops checking."""
    assert len(outbox.clean_subject("x " * 400)) <= outbox.SUBJECT_MAX


def test_clean_subject_removes_newlines():
    """A header may not contain one, and a mis-transcribed line break
    would be a header-injection shape rather than a subject."""
    got = outbox.clean_subject("lab report\nBcc: sneak@example.com")
    assert "\n" not in got


# ============================================================
# 6. The spoken lines this lane needs
# ============================================================

def test_the_failure_lines_are_distinct_and_promise_nothing_was_sent():
    assert outbox.AUTH_FAILED_LINE != outbox.WIRE_FAILED_LINE
    assert "password" in outbox.AUTH_FAILED_LINE
    for line in (outbox.AUTH_FAILED_LINE, outbox.WIRE_FAILED_LINE,
                 outbox.REHEARSAL_LINE):
        assert "nothing" in line.lower() or "Nothing" in line


def test_the_rehearsal_line_cannot_be_read_as_a_send():
    assert "Rehearsal" in outbox.REHEARSAL_LINE
    assert "nothing left the machine" in outbox.REHEARSAL_LINE.lower()


# ============================================================
# 7. The audit is written by the seam that actually sends
# ============================================================

class _RecordingSMTP:
    made: list = []

    def __init__(self, host, port, timeout=None):
        self.sent = []
        _RecordingSMTP.made.append(self)

    def login(self, user, password):
        pass

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        pass


@pytest.fixture(autouse=True)
def _no_real_sent_log(tmp_path, monkeypatch):
    """A test must never append to his real memory/sent.jsonl."""
    monkeypatch.setattr(outbox, "SENT_LOG", tmp_path / "audit.jsonl")
    _RecordingSMTP.made = []
    yield


def _sendable(tmp_path):
    root = tmp_path / "Desktop"
    root.mkdir(exist_ok=True)
    f = root / "lab_report.pdf"
    f.write_bytes(b"%PDF-1.4 body")
    st = f.stat()
    return outbox.Draft(
        path=f, size=st.st_size, mtime=st.st_mtime,
        to_addr="hjones@example.com", to_name="Heather",
        account={"label": "school", "address": "hp@tamu.edu",
                 "password": "school-secret"},
        subject="Lab report", roots=[str(root)], made_at=time.monotonic())


def test_a_real_send_writes_the_audit(tmp_path):
    line = outbox.send(_sendable(tmp_path), smtp=_RecordingSMTP)
    assert "Heather" in line
    rows = outbox.sent_rows(outbox.SENT_LOG)
    assert len(rows) == 1
    assert rows[0]["name"] == "lab_report.pdf"
    assert rows[0]["dry_run"] is False
    assert rows[0]["message_id"]


def test_the_audit_the_seam_writes_is_masked(tmp_path):
    outbox.send(_sendable(tmp_path), smtp=_RecordingSMTP)
    assert "hjones@example.com" not in \
        outbox.SENT_LOG.read_text(encoding="utf-8")


def test_a_refused_draft_writes_nothing(tmp_path):
    """Nothing left the machine, so nothing may be claimed in the audit."""
    d = _sendable(tmp_path)
    d.path.unlink()
    with pytest.raises(outbox.DraftChanged):
        outbox.send(d, smtp=_RecordingSMTP)
    assert outbox.sent_rows(outbox.SENT_LOG) == []


def test_a_failed_send_writes_nothing(tmp_path):
    class _Dead:
        def __init__(self, host, port, timeout=None):
            raise OSError("network is unreachable")

    with pytest.raises(outbox.mail_mod.MailSendFailed):
        outbox.send(_sendable(tmp_path), smtp=_Dead)
    assert outbox.sent_rows(outbox.SENT_LOG) == []


# ============================================================
# 8. The rehearsal cannot be confused with a send
# ============================================================

def test_the_env_switch_turns_a_live_send_into_a_rehearsal(tmp_path,
                                                           monkeypatch):
    """JARVIS_MAIL_DRYRUN wins over the transport it was HANDED. It can
    only ever stop a send, never cause one, so it is safe in that
    direction and only that direction."""
    monkeypatch.setenv("JARVIS_MAIL_DRYRUN", "1")
    outbox.mail_mod.DryRunSMTP.made = []
    line = outbox.send(_sendable(tmp_path), smtp=_RecordingSMTP)
    # The transport it was handed opened nothing...
    assert _RecordingSMTP.made == []
    # ...the rehearsal one assembled the message instead...
    assert len(outbox.mail_mod.DryRunSMTP.made[-1].sent) == 1
    # ...and it SAYS so, in words.
    assert line == outbox.REHEARSAL_LINE
    assert line != outbox.SENT_LINE.format(who="Heather")


def test_a_rehearsal_is_recorded_as_a_rehearsal(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_MAIL_DRYRUN", "1")
    outbox.send(_sendable(tmp_path), smtp=_RecordingSMTP)
    rows = outbox.sent_rows(outbox.SENT_LOG)
    assert len(rows) == 1 and rows[0]["dry_run"] is True
    # ...and it is therefore not in the day's sends.
    assert outbox.sent_today_line(path=outbox.SENT_LOG) == "Nothing today, sir."


def test_without_the_switch_nothing_is_a_rehearsal(tmp_path):
    line = outbox.send(_sendable(tmp_path), smtp=_RecordingSMTP)
    assert line != outbox.REHEARSAL_LINE
    assert len(_RecordingSMTP.made[-1].sent) == 1


@pytest.mark.parametrize("subject", [
    "Lab report",
    "Week nine",
    "Meeting at 3",
    "Notes from the lecture at Zachry",
    "Re: undergraduate advising",
])
def test_an_ordinary_subject_is_not_mangled_by_the_mask(tmp_path, subject):
    """The mask runs over the subject because a subject can carry an
    address. It must not chew up the ordinary ones -- the audit is read
    by eye, and "M… at 3" would make it useless."""
    log = tmp_path / "sent.jsonl"
    outbox.record_sent(_draft(tmp_path, subject=subject), path=log)
    assert outbox.sent_rows(log)[0]["subject"] == subject
