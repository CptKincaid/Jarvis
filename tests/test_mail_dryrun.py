"""The rehearsal transport, and telling a refused password from a dead wire.

Carried from the email-a-file branch (7ae1448). Two separate honesty
problems live here:

*   Before this, EVERY SMTP exception collapsed into
    ``MailSendFailed(type(exc).__name__)``, so an app password that had
    been revoked sounded exactly like the network being down -- and only
    one of those is his to fix.
*   The whole send chain (phrase -> path, name -> address, read-back,
    yes, assembled MIME) could otherwise only be proved by sending a
    real message to a real person. There is no undo for a rehearsal that
    turns out to have been live, so the dry run must be impossible to
    confuse with a send.
"""
import smtplib

import pytest

from jarvis.tools import mail as mail_mod


def _account():
    return {"label": "school", "address": "hp@tamu.edu",
            "password": "school-secret", "host": "imap.gmail.com"}


# ============================================================
# 1. A refused password is its own failure
# ============================================================

class _AuthRefusingSMTP:
    """Logs in the way a rotated app password does: it refuses."""

    def __init__(self, host, port, timeout=None):
        self.quit_called = False

    def login(self, user, password):
        raise smtplib.SMTPAuthenticationError(
            535, b"5.7.8 Username and Password not accepted")

    def send_message(self, msg):                     # pragma: no cover
        raise AssertionError("must not be reached after a refused login")

    def quit(self):
        self.quit_called = True


class _DeadWireSMTP:
    def __init__(self, host, port, timeout=None):
        raise OSError("network is unreachable")


def test_a_refused_password_raises_the_auth_failure():
    with pytest.raises(mail_mod.MailAuthFailed):
        mail_mod.send_message(_account(), "h@example.com", "s", "b",
                              smtp=_AuthRefusingSMTP)


def test_the_auth_failure_is_still_a_send_failure():
    """Callers that only know MailSendFailed must keep catching it."""
    assert issubclass(mail_mod.MailAuthFailed, mail_mod.MailSendFailed)


def test_a_dead_wire_is_NOT_an_auth_failure():
    """The whole point: these two must not sound the same."""
    with pytest.raises(mail_mod.MailSendFailed) as caught:
        mail_mod.send_message(_account(), "h@example.com", "s", "b",
                              smtp=_DeadWireSMTP)
    assert not isinstance(caught.value, mail_mod.MailAuthFailed)


def test_the_auth_failure_never_carries_the_servers_reply():
    """SMTPAuthenticationError quotes the username back, and this text
    travels into a spoken line."""
    with pytest.raises(mail_mod.MailAuthFailed) as caught:
        mail_mod.send_message(_account(), "h@example.com", "s", "b",
                              smtp=_AuthRefusingSMTP)
    said = str(caught.value)
    assert "hp@tamu.edu" not in said
    assert "Password not accepted" not in said


# ============================================================
# 2. The rehearsal opens no socket
# ============================================================

def test_dryrun_is_off_unless_the_env_says_so():
    assert mail_mod.dryrun_enabled({}) is False
    assert mail_mod.dryrun_enabled({mail_mod.DRYRUN_ENV: ""}) is False
    assert mail_mod.dryrun_enabled({mail_mod.DRYRUN_ENV: "0"}) is False


@pytest.mark.parametrize("word", ["1", "true", "TRUE", "yes", "on", " on "])
def test_dryrun_reads_the_usual_yes_words(word):
    assert mail_mod.dryrun_enabled({mail_mod.DRYRUN_ENV: word}) is True


def test_the_live_transport_is_None_not_a_real_class():
    """dryrun_transport must never be the thing that PICKS a live
    transport -- None leaves the default exactly where it was."""
    assert mail_mod.dryrun_transport({}) is None
    assert mail_mod.dryrun_transport(
        {mail_mod.DRYRUN_ENV: "1"}) is mail_mod.DryRunSMTP


def test_a_dry_run_sends_nothing(tmp_path):
    f = tmp_path / "lab_report.pdf"
    f.write_bytes(b"%PDF-1.4 body")
    mail_mod.DryRunSMTP.made = []
    mid = mail_mod.send_message(_account(), "heather@example.com",
                                "Lab report", "Sent from Jarvis.",
                                attachment=f,
                                smtp=mail_mod.dryrun_transport(
                                    {mail_mod.DRYRUN_ENV: "1"}))
    conn = mail_mod.DryRunSMTP.made[-1]
    # It assembled a real message -- the chain was genuinely exercised.
    assert mid and len(conn.sent) == 1
    assert conn.sent[0]["To"] == "heather@example.com"
    # ...and it is flagged as a rehearsal, unmistakably.
    assert mail_mod.is_dryrun(mail_mod.DryRunSMTP) is True
    assert conn.dry_run is True


def test_a_dry_run_never_stores_the_password():
    """A rehearsal must not put an app password anywhere a later print
    could reach."""
    mail_mod.DryRunSMTP.made = []
    mail_mod.send_message(_account(), "h@example.com", "s", "b",
                          smtp=mail_mod.DryRunSMTP)
    conn = mail_mod.DryRunSMTP.made[-1]
    assert conn.logged_in == ("hp@tamu.edu", "<not stored>")
    assert "school-secret" not in repr(conn.logged_in)


def test_a_real_transport_is_not_a_dry_run():
    """is_dryrun is what the spoken line reads, so a live send must
    never answer True."""
    assert mail_mod.is_dryrun(smtplib.SMTP_SSL) is False
    assert mail_mod.is_dryrun(None) is False
