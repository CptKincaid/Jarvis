"""ROUND 3: a cast may not say it landed until it has.

Every launcher, wait, settle, probe and timer here is an INJECTED
RECORDER. NOTHING starts a RustDesk session, opens a process, a socket, a
window, a camera or a capture device, and nothing touches the running
Jarvis or his desktop.
"""
from __future__ import annotations

from jarvis import castview as cv


class Clock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def subj(at=1000.0):
    return cv.view_subject(cv.SPARK, at=at)


def hp_subj(at=1000.0):
    return cv.view_subject(cv.HPCOMPUTER, at=at)


class Later:
    """The injected scheduler. It runs nothing until a test says so."""

    def __init__(self):
        self.jobs = []

    def __call__(self, delay_s, fn):
        self.jobs.append((float(delay_s), fn))
        return self

    def fire(self):
        jobs, self.jobs = list(self.jobs), []
        for _d, fn in jobs:
            fn()
        return len(jobs)


# =============================== the Spark viewer: UP is not CONNECTED
def test_a_viewer_that_is_up_but_never_connected_is_not_a_landing():
    """The module's own docstring predicts this: without direct-IP access
    and an unattended password on HPCOMPUTER a direct-IP connect WAITS for
    someone to accept it, and a RustDesk viewer sitting on that prompt is a
    live process. ``Popen.poll() is None`` answers yes and 53537c7 said
    "HPCOMPUTER's screen is on the Spark, sir." with nothing cast."""
    sink = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                            alive=lambda: True,          # the process lives
                            connected=lambda: False,     # nothing is flowing
                            settle=lambda s: None, later=Later(), now=Clock())
    res = sink.deliver(subj())
    print("\n  process alive, no stream: %r" % res.spoken)
    assert not res.landed and res.held, res
    assert sink.state.live == "", "the deck must be given back"


def test_a_sink_that_cannot_tell_says_so_instead_of_claiming_a_landing():
    """An honest "I cannot tell" is a correct outcome and is better than a
    confident wrong sentence. Two shapes of it: no probe wired at all, and
    a probe that has no opinion this time."""
    blind = cv.SparkViewSink(launch=lambda h: None, alive=lambda: True,
                             settle=lambda s: None, now=Clock())
    ok, why = blind.available()
    print("\n  no connection probe wired: available=%s (%s)" % (ok, why))
    assert not ok and why == cv.NO_CONNECTION_PROBE_REASON
    res = blind.deliver(subj())
    assert not res.landed and res.held

    dunno = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                             alive=lambda: True, connected=lambda: None,
                             settle=lambda s: None, later=Later(), now=Clock())
    res2 = dunno.deliver(subj())
    print("  probe has no opinion    : %r" % res2.spoken)
    assert not res2.landed and res2.held
    assert dunno.state.live == ""


def test_a_viewer_that_dies_after_the_settle_gives_the_deck_back():
    """MEASURED at 53537c7: a viewer alive at the 0.7 s settle and dead at
    3 s was claimed landed, NOTHING ever re-checked it, and ``state.live``
    stayed "spark-view" -- so the deck was held by a cast that was not
    there and the next genuine cast was refused as busy."""
    up = {"alive": True, "conn": True}
    said = []
    later = Later()
    sink = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                            alive=lambda: up["alive"],
                            connected=lambda: up["conn"],
                            settle=lambda s: None, later=later,
                            retract=said.append, now=Clock())
    res = sink.deliver(subj())
    assert res.landed and sink.state.live == "spark-view"
    up["alive"] = False                       # it dies at three seconds
    fired = later.fire()
    print("\n  re-checks scheduled: %d" % fired)
    print("  after it dies: state.live=%r, said %r" % (sink.state.live, said))
    assert fired == 1
    assert sink.state.live == "", "a cast that died must give the deck back"
    assert said and cv.CAST_GONE_LINE in said[0]


def test_a_viewer_that_stays_up_keeps_the_deck_and_says_nothing_more():
    said = []
    later = Later()
    sink = cv.SparkViewSink(launch=lambda h: None, stop=lambda: None,
                            alive=lambda: True, connected=lambda: True,
                            settle=lambda s: None, later=later,
                            retract=said.append, now=Clock())
    res = sink.deliver(subj())
    assert res.landed
    later.fire()
    assert sink.state.live == "spark-view" and said == []


# ================================== the ack wait: one receipt, one waiter
def test_an_unrelated_receipt_does_not_spend_the_whole_ack_wait():
    """MEASURED at 53537c7: waiting on seq 2 and woken by the receipt for
    seq 1, ``await_ack`` reported NO RECEIPT and held -- it calls a genuine
    cast failed. The wait must be for OUR sequence."""
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    relay.set_verb(cv.VERB_STOP)                      # seq 1, unacked
    seq2 = relay.set_verb(cv.VERB_SHOW)               # seq 2, the one we want
    rounds = []

    def wait(timeout_s):
        """A faithful stand-in for ``threading.Event.wait``: it returns
        True when it was woken and False when it timed out."""
        rounds.append(timeout_s)
        if len(rounds) == 1:
            relay.poll(1, timeout_s=0.0)              # the OLD verb is acked
            return True                               # ...and it wakes us
        relay.poll(seq2, timeout_s=0.0)               # now OURS arrives
        return True
    got = relay.await_ack(seq2, 2.0, wait=wait)
    print("\n  waits entered: %d, acked through %d, receipt=%s"
          % (len(rounds), relay.acked, got))
    assert got is True, "the receipt for seq 2 did arrive"
    assert len(rounds) >= 2


def test_the_ack_wait_is_bounded_and_still_gives_up():
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    seq = relay.set_verb(cv.VERB_SHOW)
    calls = []

    def never(timeout_s):
        calls.append(timeout_s)
        return True                     # woken every time, never by our seq
    assert relay.await_ack(seq, 2.0, wait=never) is False
    print("\n  bounded at %d rounds" % len(calls))
    assert 1 <= len(calls) <= cv.ACK_WAIT_ROUNDS


# ======================= the Windows helper says the launch failed
def test_the_helper_can_report_that_the_launch_failed():
    """The shipped script committed the receipt BEFORE ``Start-Process``
    and swallowed the error, so Jarvis said "The Spark's screen is on
    HPCOMPUTER, sir." with 1 launch attempt and 0 windows up. The helper
    must be able to say it could not open the viewer, and Jarvis must
    repeat that rather than claim a landing."""
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    relay.note(mon=1, layout="0,1920", at=clk.t)
    waits = []

    def wait(timeout_s):
        waits.append(timeout_s)
        # the helper answers, but with a failure and NO receipt
        relay.note(mon=1, layout="0,1920", at=clk.t,
                   fail="no-viewer", failseq=relay.seq)
        return True
    sink = cv.HpViewSink(relay=relay, ack_s=2.0, wait=wait, now=clk)
    res = sink.deliver(hp_subj())
    print("\n  helper reported %r -> %r" % (relay.fail, res.spoken))
    assert not res.landed and res.held
    assert cv.LAUNCH_FAILED_LINE in res.spoken
    assert relay.verb == cv.VERB_STOP
    assert sink.state.live == ""


def test_a_failure_code_from_the_helper_is_a_closed_set_too():
    relay = cv.CastRelay(now=Clock())
    relay.note(mon=1, layout="0,1920", fail="rm -rf /", failseq=1)
    print("\n  a failure that is not in the set: %r" % relay.fail)
    assert relay.fail == ""
    relay.note(mon=1, layout="0,1920", fail="launch-failed", failseq=1)
    assert relay.fail == "launch-failed"


# ============================== the shipped PowerShell, as a faithful model
class ShippedHelper:
    """A faithful model of scripts/windows/cast-poll.ps1 AS IT NOW IS.

    The order is read off the file by
    ``test_the_script_commits_its_receipt_only_after_a_verified_launch``;
    this models what that order does over a round trip.
    """

    def __init__(self, relay, *, launch_works=True, binary_there=True):
        self.relay = relay
        self.seq = -1
        self.fail_seq = -2
        self.fail = ""
        self.launch_works = launch_works
        self.binary_there = binary_there
        self.windows_up = 0
        self.launch_attempts = 0

    def poll_once(self, at=None):
        self.relay.note(mon=1, layout="0,1920,1920,1920", at=at,
                        fail=self.fail, failseq=self.fail_seq)
        out = self.relay.poll(self.seq, timeout_s=0.0)
        new, verb = int(out["seq"]), str(out["verb"])
        if new != self.seq and new != self.fail_seq:
            ok = True
            if verb == cv.VERB_SHOW:
                ok = False
                if not self.binary_there:
                    self.fail = "no-viewer"
                else:
                    self.launch_attempts += 1
                    if self.launch_works:
                        self.windows_up += 1
                        ok = True
                    else:
                        self.fail = "viewer-exited"
            elif verb == cv.VERB_STOP:
                self.windows_up = 0
            if ok:
                self.seq = new                  # <-- only now
                self.fail = ""
            else:
                self.fail_seq = new
        return out

    def round_trip(self, at=None):
        self.poll_once(at=at)
        self.poll_once(at=at)


def test_a_launch_that_fails_never_becomes_a_receipt():
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    helper = ShippedHelper(relay, binary_there=False)
    helper.poll_once(at=clk.t)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                         wait=lambda s: (helper.round_trip(at=clk.t), True)[1],
                         now=clk)
    res = sink.deliver(hp_subj())
    print("\n  %d launch attempts, %d windows up -> %r"
          % (helper.launch_attempts, helper.windows_up, res.spoken))
    assert helper.windows_up == 0
    assert not res.landed and res.held


def test_a_launch_that_works_still_becomes_a_receipt():
    clk = Clock()
    relay = cv.CastRelay(now=clk)
    helper = ShippedHelper(relay, launch_works=True)
    helper.poll_once(at=clk.t)
    sink = cv.HpViewSink(relay=relay, ack_s=2.0,
                         wait=lambda s: (helper.round_trip(at=clk.t), True)[1],
                         now=clk)
    res = sink.deliver(hp_subj())
    print("\n  launch works -> %r (windows up %d)"
          % (res.spoken, helper.windows_up))
    assert res.landed and helper.windows_up == 1
