"""'What was my last email about?' must search read mail too.

get_mail defaults to UNSEEN. Its parameter description already tells the model
that unread_only=false is "what answers 'my last email'" -- and the local model
still sent true, so at 21:03 on 2026-08-29 every account was searched UNSEEN
(personal 5, work 0, school 4) and the newest READ messages were invisible.
Jarvis named a message from that morning as the latest when newer mail existed.

Documenting it harder was already tried, so the flag is pinned in the router
instead of asked for. The model still renders the prose; it just no longer
chooses whether to look.
"""
from types import SimpleNamespace

import pytest

from jarvis.commander import _LAST_MAIL_RX, _h_last_mail


@pytest.mark.parametrize("said", [
    "what was my last email about?",
    "my latest email",
    "read me my most recent email",
    "what's my newest mail",
    "tell me about my last e-mail",
    "what did mark say in his last email",       # a name, not a verb
    "what was my last email about the move",     # a noun, not a verb
])
def test_the_phrasings_that_mean_most_recent(said):
    assert _LAST_MAIL_RX.search(said), said


@pytest.mark.parametrize("said", [
    "do i have any new email",
    "any unread mail?",
    "check my email",
    "send an email to mum",
    "what's in my inbox",
    "what was the last message you sent",     # discord / notes / session, not mail
    "read my latest message",
])
def test_unread_and_send_queries_are_left_alone(said):
    """These genuinely want the UNSEEN default, or are not a read at all."""
    assert not _LAST_MAIL_RX.search(said), said


def test_the_flag_is_pinned_not_suggested():
    calls = []
    brain = SimpleNamespace(chat=lambda t, **kw: calls.append((t, kw)))
    c = SimpleNamespace(_svc=lambda name: brain if name == "brain" else None)

    said = "what was my last email about?"
    res = _h_last_mail(c, said, _LAST_MAIL_RX.search(said))

    assert res is not None and res.handled and res.done is False
    (text, kw), = calls
    assert text == said
    assert kw["force_tool"] == "get_mail"
    assert kw["force_args"]["unread_only"] is False, "still searching UNSEEN"
    assert kw["force_args"]["limit"] == 1


def test_it_looks_back_further_than_a_day():
    """"My last email" is not "since midnight" -- the 24 h default would
    report nothing on a quiet morning."""
    calls = []
    brain = SimpleNamespace(chat=lambda t, **kw: calls.append(kw))
    c = SimpleNamespace(_svc=lambda name: brain)
    _h_last_mail(c, "my latest email", None)
    hours = calls[0]["force_args"]["since_hours"]
    assert 24 < hours <= 336, hours          # 336 is the schema maximum


def test_no_brain_means_no_claim_to_have_handled_it():
    c = SimpleNamespace(_svc=lambda name: None)
    assert _h_last_mail(c, "my last email", None) is None


@pytest.mark.parametrize("said", [
    "what did mark say in his last email",
    "what was my last email about the move",
])
def test_reads_that_merely_contain_a_write_word_still_read(said):
    """The first guard was a bare word list, so a sender called Mark or a
    subject about a move made the read fall through to the model."""
    calls = []
    brain = SimpleNamespace(chat=lambda t, **kw: calls.append(kw))
    c = SimpleNamespace(_svc=lambda name: brain)
    assert _h_last_mail(c, said, None) is not None, said
    assert calls and calls[0]["force_tool"] == "get_mail"


@pytest.mark.parametrize("said", [
    "reply to my latest email", "delete my last email",
    "could you forward the last mail to bob", "please archive my most recent email",
])
def test_leading_write_verbs_fall_through(said):
    c = SimpleNamespace(_svc=lambda name: SimpleNamespace(chat=lambda *a, **k: None))
    assert _h_last_mail(c, said, None) is None, said
