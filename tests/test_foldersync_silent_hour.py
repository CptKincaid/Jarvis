"""THE PARKED HOUR.  He can see the file. Jarvis says nothing for an hour.

A file that fails :data:`MAX_ATTEMPTS` times is parked for
:data:`RETRY_AFTER_S` -- 5 tries, then 3600 seconds.  That is the right
BEHAVIOUR: retrying a broken thing every 30 seconds for an hour is how a
lane spends a morning failing loudly at one file while the queue behind it
waits.  Nothing here changes the parking.

What was wrong is that the park was a bare ``continue``.  Both sites:

    if self.ledger.blocked(f"pull:{entry.key}", now):
        continue                       # no event, no status line, nothing
    if self.ledger.blocked(f"push:{p.name}|{stat_key(p)}", now):
        continue                       # the same, outbound

So for a full hour, once every 30 seconds, this lane looked at a file, made
a decision about it, and wrote down nothing.  On the inbound side he can see
that file sitting in the HPCOMPUTER outbox the entire time.  On the outbound
side status.txt lists it under "outbox N waiting" and never says why it is
still waiting.  Round 5 already ruled on this exact shape three times over
-- the unsafe inbound name, the remote folder, the SKIP_ prefix -- under the
heading THE THREE SILENCES, said out loud.  This is the fourth, and it is
the one with a clock on it.

His words about a different silence the same night: a thing he can see with
his own eyes while Jarvis says nothing is the defect.

NOTHING HERE OPENS A SOCKET OR TOUCHES ~/Desktop.  tmp_path and the fake
transport from tests/test_foldersync.py, which is where the parking rule is
already tested for its timing (test_a_file_that_fails_five_times_is_parked).
"""
import time

import pytest

from jarvis import foldersync as fs
from tests.test_foldersync import FakeTransport, drop, home, syncer  # noqa: F401


def notes(sync) -> str:
    """status.txt ABOVE the "recent" block.

    The history block quotes the name and the reason of every recent event,
    so a test that greps the whole file passes on the record of the five
    FAILURES and proves nothing about the hour of silence that follows
    them.  Three of the assertions below passed that way on the first
    draft.  What is under test is what this pass SAYS, so cut the history
    off.
    """
    return sync.status_text().split("\nrecent\n")[0]


def _park_inbound(sync, tr, name="broken.pdf", data=b"x" * 40):
    """Fail the same inbound file MAX_ATTEMPTS times, which parks it.

    Returns the moment the last attempt happened, so a caller can ask what
    the next pass says while the hour is still running.
    """
    tr.put("outbox", name, data)
    tr.fetch_fail[name] = "failed"
    now = 1000.0
    for _ in range(fs.MAX_ATTEMPTS):
        sync.pull_once(now)
        now += 60.0
    return now


def test_the_setup_is_real_a_file_really_is_parked(home):
    """Guard the guard.  If this stops parking, every test below is testing
    nothing and would pass for the wrong reason."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    now = _park_inbound(sync, tr)
    key = [e.key for e in tr.listing("outbox")[0] if e.name == "broken.pdf"][0]
    assert sync.ledger.blocked(f"pull:{key}", now), "nothing was parked"
    assert not sync.ledger.blocked(f"pull:{key}", now + fs.RETRY_AFTER_S), \
        "the park never expires"


def test_a_parked_inbound_file_gets_a_line_in_status(home):
    """THE DEFECT.  He is looking at the file on HPCOMPUTER; status.txt must
    say it is not coming yet, and when it will be tried again."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    now = _park_inbound(sync, tr)

    sync.pull_once(now + 30.0)
    text = notes(sync)
    assert "broken.pdf" in text, (
        "a file he can see in the HPCOMPUTER outbox is skipped every pass "
        "for an hour and status.txt does not name it:\n" + text)
    assert "stopped retrying" in text, text


def test_the_status_line_says_when_it_will_be_tried_again(home):
    """"I have stopped" without "and I start again at" is half a sentence.
    An hour is long enough that the difference is the whole point."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    now = _park_inbound(sync, tr)
    sync.pull_once(now + 30.0)
    text = notes(sync)
    again = time.strftime("%H:%M:%S",
                          time.localtime(now - 60.0 + fs.RETRY_AFTER_S))
    assert again in text, (
        f"status.txt does not say when it tries again (expected {again}):\n"
        + text)


def test_a_parked_inbound_file_gets_an_event(home):
    """status.txt is rewritten every pass and keeps no history.  The record
    is what he has afterwards, so the pass that decided to skip must leave
    a line in it too."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    now = _park_inbound(sync, tr)

    events = sync.pull_once(now + 30.0)
    parked = [e for e in events if e.name == "broken.pdf"]
    assert parked, (
        "the pass skipped a file and recorded no event at all: "
        f"{[(e.name, e.outcome) for e in events]}")
    assert parked[0].outcome == "parked", parked[0].outcome
    assert parked[0].direction == "pull"


def test_a_parked_outbound_file_gets_a_line_too(home):
    """The push side has the identical bare ``continue``.  His file is in
    his own Outbox, listed under "outbox 1 waiting", with nothing anywhere
    saying why it has stopped moving."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    p = drop(home, "stuck.txt", b"y" * 30)
    tr.send_fail["stuck.txt"] = "failed"
    now = 1000.0
    for _ in range(fs.MAX_ATTEMPTS):
        sync.push_once(now)
        now += 60.0
    assert p.exists(), "his file was moved while failing"

    sync.push_once(now + 30.0)
    text = notes(sync)
    assert "stuck.txt" in text and "stopped retrying" in text, (
        "his own Outbox file is parked for an hour and status.txt says only "
        "that it is waiting:\n" + text)


def test_the_line_goes_when_the_hour_is_up(home):
    """THIS PASS'S EVIDENCE, never the last pass's.  A note that outlives
    the thing it describes is the ``_foreign_temps`` mistake, which this
    lane has already made once."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    now = _park_inbound(sync, tr)
    sync.pull_once(now + 30.0)
    assert "stopped retrying" in notes(sync)

    tr.fetch_fail.clear()                       # whatever it was, it is fixed
    sync.pull_once(now + fs.RETRY_AFTER_S + 1.0)
    text = notes(sync)
    assert "stopped retrying" not in text, (
        "the parked note survived the park:\n" + text)


def test_the_parked_line_never_claims_the_link_is_down(home):
    """A parked FILE says nothing about HPCOMPUTER, which is answering every
    pass.  Finding K was exactly this confusion pointed the other way, and
    it took his sync interval from 30 s to 300 s while the box was healthy.
    """
    tr = FakeTransport()
    sync = syncer(home, tr)
    now = _park_inbound(sync, tr)
    sync.pull_once(now + 30.0)
    text = notes(sync)
    assert "stopped retrying" in text, text
    assert "link      DOWN" not in text, text
    assert sync._down_reason == "", sync._down_reason


def test_a_parked_file_does_not_stop_the_queue_behind_it(home):
    """The reason parking exists at all.  Pinned here so the new line
    cannot be added by removing the behaviour it describes."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    now = _park_inbound(sync, tr)
    tr.put("outbox", "good.pdf", b"z" * 12)

    sync.pull_once(now + 30.0)
    assert (home / "Inbox" / "good.pdf").exists(), (
        "the parked file blocked the one behind it")


@pytest.mark.parametrize("reason", ["failed", "denied", "verify-failed"])
def test_the_line_carries_the_reason_it_stopped(home, reason):
    """Five failures for "denied" and five for a size mismatch are different
    problems with different fixes, and the ledger already stores which."""
    tr = FakeTransport()
    sync = syncer(home, tr)
    tr.put("outbox", "broken.pdf", b"x" * 40)
    tr.fetch_fail["broken.pdf"] = reason
    now = 1000.0
    for _ in range(fs.MAX_ATTEMPTS):
        sync.pull_once(now)
        now += 60.0
    sync.pull_once(now + 30.0)
    assert reason in notes(sync), notes(sync)
